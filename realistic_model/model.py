# NYC tract SEIR + vaccination model with commuting between tracts, fitted to NYC reported cases.
# Day 0 = 2020-02-29. Runs to 2021-11-30 (day 640), before Omicron (immune escape is not modelled).
# Transmission = citywide level b(t) fitted every 2 weeks x a relative factor per ZIP (MODZCTA) and period,
# fitted to DOHMH weekly ZIP case rates in the pre-vaccine periods only. The fit uses the reconstructed actual rollout.
import os, sys, numpy as np, pandas as pd, geopandas as geopd, scipy.sparse as sp
from scipy.optimize import minimize_scalar

SCR = 'realistic_model'   # data and outputs; run scripts from the repo root
DAY0 = pd.Timestamp('2020-02-29'); END = 640
SIGMA, GAMMA = 1/4, 1/6        # latent 4 days, infectious 6 days (generation time ~7 days)
COMMUTE_SHARE = 1/3            # share of a commuter's contacts that happen in the work tract
SERO_DAY, SERO = 51, 0.227     # 22.7% of NYC infected by ~Apr 20 2020 (NYS antibody survey, Rosenberg et al. 2020)
RHO_LATE = float(os.environ.get('RHO_LATE', 0.5))  # share of infections reported from Jun 2020
PERIODS = [0, 92, 276, 457, 549, END]   # Mar-May 2020, Jun-Nov 2020, Dec 2020-May 2021, Jun-Aug 2021, Sep-Nov 2021
DT = 0.5; WINDOW = 14
# ZIP factors are fitted on Mar-Nov 2020 only; the Jun-Nov 2020 factors are kept for later periods, so they do not
# absorb the effects of the actual rollout. Only the citywide b(t) is fitted after.
LAST_FITTED_PERIOD = 1
FIT_FILE = f'fit3_rho{RHO_LATE}.pkl'

# ---------- tracts ----------
t = geopd.read_file('nycCensusTracts/nyct2020.dbf'); t['GEOID'] = t['GEOID'].astype(np.int64)
c = pd.read_csv('datasets/nyc_census_tracts.csv')
df = pd.merge(t, c, left_on='GEOID', right_on='CensusTract')
df = df[df.TotalPop != 0].dropna(subset=['IncomePerCap']).reset_index(drop=True)
n = len(df); pop = df.TotalPop.to_numpy(float); inc = df.IncomePerCap.to_numpy()
boro = df.BoroName.to_numpy()
BOROS = {'Bronx': 'BX', 'Brooklyn': 'BK', 'Manhattan': 'MN', 'Queens': 'QN', 'Staten Island': 'SI'}
CENSUS2020 = {'Bronx': 1472654, 'Brooklyn': 2736074, 'Manhattan': 1694251, 'Queens': 2405464, 'Staten Island': 495747}
boroScale = np.array([pop[boro == b].sum() / CENSUS2020[b] for b in boro])   # model pop / real pop, per tract's borough

tz = pd.read_csv(f'{SCR}/tract_to_modzcta.csv').set_index('GEOID').MODZCTA
zipOf = df.GEOID.map(tz).to_numpy()
zips = np.unique(zipOf); nz = len(zips); zi = np.searchsorted(zips, zipOf)
Z = sp.csr_matrix((np.ones(n), (zi, np.arange(n))), shape=(nz, n))   # ZIP x tract aggregation

# ---------- commuting mixing matrix ----------
idx = {g: k for k, g in enumerate(df.GEOID)}
fl = pd.read_csv(f'{SCR}/flows.csv'); fl = fl[fl.h.isin(idx) & fl.w.isin(idx) & (fl.h != fl.w)]
F = np.zeros((n, n)); F[fl.h.map(idx), fl.w.map(idx)] = fl.S000
# ACS populations use 2010 tract IDs and LODES uses 2020 tracts, so per-tract worker/population ratios are unreliable.
# Use the citywide share of residents commuting to another tract, and LODES only for where they go.
commuterShare = F.sum() / pop.sum()
dest = np.divide(F, F.sum(1, keepdims=True), out=np.zeros_like(F), where=F.sum(1, keepdims=True) > 0)
M = COMMUTE_SHARE * commuterShare * dest
M[np.arange(n), np.arange(n)] = 1 - M.sum(1)
Npresent = M.T @ pop
M = sp.csr_matrix(M); MT = M.T.tocsr()

# ---------- cases ----------
d = pd.read_csv(f'{SCR}/daily.csv'); d['date'] = pd.to_datetime(d.date_of_interest); d = d.sort_values('date')
d = d[d.date >= DAY0].iloc[:END].reset_index(drop=True)
boroCases = {b: (d[f'{ab}_CASE_COUNT'] + d[f'{ab}_PROBABLE_CASE_COUNT']).to_numpy(float) * pop[boro == b].sum() / CENSUS2020[b]
             for b, ab in BOROS.items()}
cityCases = sum(boroCases.values())
citySmooth = pd.Series(cityCases).rolling(7, center=True, min_periods=1).mean().to_numpy()

# ZIP case rates per period, relative to the city rate (DOHMH weekly case rate per 100k)
cr = pd.read_csv(f'{SCR}/caserate_modzcta.csv'); cr['day'] = (pd.to_datetime(cr.week_ending) - DAY0).dt.days
obsRel = np.full((len(PERIODS) - 1, nz), np.nan)
for p in range(len(PERIODS) - 1):
  w = cr[(cr.day > PERIODS[p]) & (cr.day <= PERIODS[p + 1])]
  for k, zc in enumerate(zips):
    col = f'CASERATE_{int(zc)}'
    if col in w: obsRel[p, k] = w[col].sum() / w.CASERATE_CITY.sum()

# ---------- vaccination (people with 1+ dose, NYC residents, by borough) ----------
vb = pd.read_csv(f'{SCR}/vax_byboro.csv'); vb['day'] = (pd.to_datetime(vb.DATE) - DAY0).dt.days
VAB = {'Bronx': 'BX', 'Brooklyn': 'BK', 'Manhattan': 'MH', 'Queens': 'QS', 'Staten Island': 'SI'}
boroDoses = {}
for b, ab in VAB.items():
  cum = pd.to_numeric(vb[f'{ab}_COUNT_1PLUS_CUMULATIVE'], errors='coerce').fillna(0).cummax().to_numpy()
  daily = np.zeros(END); new = np.diff(np.concatenate([[0], cum])); ok = vb.day < END
  daily[vb.day[ok]] = new[ok.to_numpy()] * pop[boro == b].sum() / CENSUS2020[b]
  boroDoses[b] = daily
dosesPerDay = sum(boroDoses.values())

# within each borough, split doses across ZIPs by each ZIP's share of the borough's new vaccinations between snapshots
snap = pd.read_csv(f'{SCR}/zip_vax_snapshots.csv'); snap['day'] = (pd.to_datetime(snap.DATE) - DAY0).dt.days
snapDays = sorted(snap.day.unique())
cumZ = np.zeros((len(snapDays), nz))
for s, sd in enumerate(snapDays):
  ss = snap[snap.day == sd].set_index('MODZCTA').COUNT_1PLUS_CUMULATIVE
  cumZ[s] = [ss.get(int(zc), 0) for zc in zips]
zipBoro = np.array([pd.Series(boro[zi == k]).mode()[0] for k in range(nz)])
def zipShares(day):
  s = np.searchsorted(snapDays, day)   # interval (snapDays[s-1], snapDays[s]]
  inc_ = cumZ[0] if s == 0 else cumZ[min(s, len(snapDays) - 1)] - cumZ[min(s, len(snapDays) - 1) - 1]
  inc_ = np.maximum(inc_, 0); out = np.zeros(nz)
  for b in BOROS:
    m = zipBoro == b; out[m] = inc_[m] / inc_[m].sum()
  return out
zipShareByDay = {day: zipShares(day) for day in range(END) if dosesPerDay[day] > 0}

def efficacy(day): return 0.9 if day < 488 else 0.65   # Delta dominant from ~Jul 1 2021

def rho(day, rhoEarly):
  if day <= 61: return rhoEarly
  if day >= 92: return RHO_LATE
  return rhoEarly + (RHO_LATE - rhoEarly) * (day - 61) / 31
rhoEarly = cityCases[:SERO_DAY + 1].sum() / (SERO * pop.sum())

def periodOf(day): return np.searchsorted(PERIODS, day, side='right') - 1

# ---------- allocation strategies: return (doses, weights) for a day ----------
def actualAllocation(day, y):
  U = y[0] + y[6]; Uz = Z @ U
  zipDoses = np.zeros(nz)
  for b in BOROS: zipDoses[zipBoro == b] = boroDoses[b][day] * zipShareByDay[day][zipBoro == b]
  return dosesPerDay[day], np.divide(zipDoses, Uz, out=np.zeros(nz), where=Uz > 0)[zi]
uniformWeights = np.ones(n)
zInc = (inc - np.median(inc)) / inc.std(); incomeWeights = zInc.max() - zInc
def uniformAllocation(day, y): return dosesPerDay[day], uniformWeights
def incomeAllocation(day, y): return dosesPerDay[day], incomeWeights
# The targeted strategies below weight a tract by how far its score is above the lowest tract, as income-based does.
def susceptibleAllocation(day, y):   # share of unvaccinated residents never infected, recomputed daily
  S, R = y[0], y[6]; s = np.divide(S, S + R, out=np.zeros(n), where=S + R > 0)
  return dosesPerDay[day], s - s.min()
def hotspotAllocation(rByPeriod):   # current force of infection on residents, recomputed daily
  def allocation(day, y):
    lam = rByPeriod[periodOf(day)] * (M @ ((MT @ (y[4] + y[5])) / Npresent))
    return dosesPerDay[day], lam - lam.min()
  return allocation

# ---------- model ----------
# state rows: S, Sv, E, Ev, I, Iv, R, Rv, cumulative onsets (E->I)
def rhs(y, b, r, e):
  S, Sv, E, Ev, I, Iv, R, Rv, C = y
  lam = b * r * (M @ ((MT @ (I + Iv)) / Npresent))
  iS, iSv = lam * S, (1 - e) * lam * Sv
  return np.array([-iS, -iSv, iS - SIGMA * E, iSv - SIGMA * Ev, SIGMA * E - GAMMA * I, SIGMA * Ev - GAMMA * Iv,
                   GAMMA * I, GAMMA * Iv, SIGMA * (E + Ev)])

# Uptake cap: no ZIP vaccinates more of its residents than actually got a first dose by Dec 2021,
# so a different allocation changes when people are vaccinated, not how many are willing.
finalCoverage = np.clip(cumZ[-1] / np.array([snap[(snap.day == snapDays[-1]) & (snap.MODZCTA == int(zc))].POP_DENOMINATOR.sum() or np.nan for zc in zips]), 0, 0.99)
willing = np.nan_to_num(finalCoverage, nan=np.nanmean(finalCoverage))[zi] * pop

def vaccinate(y, doses, weights):
  S, Sv, E, Ev, I, Iv, R, Rv, C = y
  remaining = doses
  while remaining > 1e-6:
    U = S + R
    U = np.minimum(U, np.maximum(willing - (Sv + Ev + Iv + Rv), 0))
    demand = weights * U
    if demand.sum() <= 1e-6: demand = U.copy()
    if demand.sum() <= 1e-6: break
    given = np.minimum(remaining * demand / demand.sum(), U)
    fracS = np.divide(S, S + R, out=np.zeros(n), where=S + R > 0)
    S, Sv = S - given * fracS, Sv + given * fracS
    R, Rv = R - given * (1 - fracS), Rv + given * (1 - fracS)
    remaining -= given.sum()
    weights = np.where(given < U - 1e-9, weights, 0)   # full tracts drop out; leftovers go to the rest
  return np.array([S, Sv, E, Ev, I, Iv, R, Rv, C])

def step_days(y, day0, day1, b, rByPeriod, allocation):
  daily = [y[8].copy()]
  for day in range(day0, day1):
    if allocation is not None and dosesPerDay[day] > 0: y = vaccinate(y, *allocation(day, y))
    e = efficacy(day); r = rByPeriod[periodOf(day)]
    for _ in range(int(1 / DT)):
      k1 = rhs(y, b, r, e); k2 = rhs(y + DT/2*k1, b, r, e); k3 = rhs(y + DT/2*k2, b, r, e); k4 = rhs(y + DT*k3, b, r, e)
      y = y + DT/6 * (k1 + 2*k2 + 2*k3 + k4)
    daily.append(y[8].copy())
  return y, np.diff(np.array(daily), axis=0)

def initial(seed, rByPeriod):
  E0 = seed * pop * rByPeriod[0] / (pop * rByPeriod[0]).sum()
  y = np.zeros((9, n)); y[0] = pop - E0; y[2] = E0
  return y

def tractFactors(zipFactors):
  r = zipFactors[:, zi]
  return r / (r * pop).sum(1, keepdims=True) * pop.sum()

def fitCitywide(rByPeriod, allocation, seeds=(1000, 3000, 10000), verbose=False):
  def loss(days, onsets):
    model = onsets.sum(1) * np.array([rho(dd, rhoEarly) for dd in days])
    return np.sum((np.log(model + 1) - np.log(citySmooth[days] + 1)) ** 2)
  best = None
  for seed in seeds:   # first window: fit seed and transmission together
    y0 = initial(seed, rByPeriod)
    res = minimize_scalar(lambda lb: loss(range(0, WINDOW), step_days(y0, 0, WINDOW, np.exp(lb), rByPeriod, allocation)[1]),
                          bounds=(np.log(0.02), np.log(1.0)), method='bounded', options={'xatol': 1e-2})
    if best is None or res.fun < best[0]: best = (res.fun, seed)
  seed = best[1]; y = initial(seed, rByPeriod); bs = []; allOnsets = []
  for start in range(0, END, WINDOW):
    stop = min(start + WINDOW, END); days = range(start, stop)
    res = minimize_scalar(lambda lb: loss(days, step_days(y, start, stop, np.exp(lb), rByPeriod, allocation)[1]),
                          bounds=(np.log(0.02), np.log(1.0)), method='bounded', options={'xatol': 1e-2})
    b = np.exp(res.x); bs.append(b)
    y, on = step_days(y, start, stop, b, rByPeriod, allocation); allOnsets.append(on)
    if verbose: print(f'  day {start:3d} {(DAY0 + pd.Timedelta(days=start)).date()} R~{b/GAMMA:.2f}', flush=True)
  return dict(seed=seed, bs=np.array(bs), onsets=np.concatenate(allOnsets), final=y)

def modelRel(onsets):
  rep = onsets * np.array([rho(dd, rhoEarly) for dd in range(len(onsets))])[:, None]
  out = np.zeros((len(PERIODS) - 1, nz))
  for p in range(len(PERIODS) - 1):
    perCap = (Z @ rep[PERIODS[p]:PERIODS[p + 1]].sum(0)) / (Z @ pop)
    out[p] = perCap / (rep[PERIODS[p]:PERIODS[p + 1]].sum() / pop.sum())
  return out

def simulate(fitResult, rByPeriod, allocation):
  y = initial(fitResult['seed'], rByPeriod); allOnsets = []
  for k, start in enumerate(range(0, END, WINDOW)):
    y, on = step_days(y, start, min(start + WINDOW, END), fitResult['bs'][k], rByPeriod, allocation); allOnsets.append(on)
  return np.concatenate(allOnsets), y
