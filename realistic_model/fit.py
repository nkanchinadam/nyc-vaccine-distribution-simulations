# Fits ZIP transmission factors period by period (later periods cannot affect earlier ones) up to LAST_FITTED_PERIOD.
import sys, pickle
exec(open(sys.argv[1]).read())
ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 5
P = len(PERIODS) - 1; has = ~np.isnan(obsRel); zpop = Z @ pop

def lossFn(days, onsets):
  model = onsets.sum(1) * np.array([rho(dd, rhoEarly) for dd in days])
  return np.sum((np.log(model + 1) - np.log(citySmooth[list(days)] + 1)) ** 2)

def fitWindows(y, start, stop, rr, alloc):
  windows, ons = [], []
  for ws in range(start, stop, WINDOW):
    we = min(ws + WINDOW, stop); days = range(ws, we)
    res = minimize_scalar(lambda lb: lossFn(days, step_days(y, ws, we, np.exp(lb), rr, alloc)[1]),
                          bounds=(np.log(0.02), np.log(1.0)), method='bounded', options={'xatol': 1e-2})
    b = np.exp(res.x); y, on = step_days(y, ws, we, b, rr, alloc); windows.append((ws, we, b)); ons.append(on)
  return y, windows, np.concatenate(ons)

def periodRel(on, p):
  rep = on * np.array([rho(dd, rhoEarly) for dd in range(PERIODS[p], PERIODS[p + 1])])[:, None]
  return (Z @ rep.sum(0)) / zpop / (rep.sum() / pop.sum())

def err(mr, p): return np.sqrt(np.average(np.log(obsRel[p, has[p]] / mr[has[p]]) ** 2, weights=zpop[has[p]]))

zf = np.ones((P, nz)); allWindows, allOnsets = [], []
for p in range(P):
  # damped updates: if the error grows, go back to the best factors and halve the step
  stepZ = 0.4; best = None
  frozen = p > LAST_FITTED_PERIOD
  if frozen: zf[p] = zf[LAST_FITTED_PERIOD]
  for it in range(1 if frozen else ITER + 1):
    rr = tractFactors(zf)
    if p == 0:   # choose the initial seed size together with the first window
      seedBest = None
      for seed in (1000, 3000, 10000):
        y0 = initial(seed, rr)
        r_ = minimize_scalar(lambda lb: lossFn(range(0, WINDOW), step_days(y0, 0, WINDOW, np.exp(lb), rr, None)[1]),
                             bounds=(np.log(0.02), np.log(1.0)), method='bounded', options={'xatol': 1e-2})
        if seedBest is None or r_.fun < seedBest[0]: seedBest = (r_.fun, seed)
      seed = seedBest[1]; yStart = initial(seed, rr)
    y, windows, on = fitWindows(yStart, PERIODS[p], PERIODS[p + 1], rr, actualAllocation)
    mr = periodRel(on, p); e = err(mr, p)
    print(f'period {p} iter {it}: ZIP err {e:.3f} step {stepZ:.3f} max R {max(w[2] for w in windows)/GAMMA:.2f}', flush=True)
    if best is None or e < best['e']:
      best = dict(e=e, zf=zf[p].copy(), y=y, windows=windows, on=on, seed=seed if p == 0 else None)
    if best['e'] < e: stepZ /= 2   # error went up: step again from the best factors, with half the step
    else: best['mr'] = mr
    if it < ITER and not frozen:
      zf[p] = np.clip(best['zf'] * np.where(has[p], obsRel[p] / np.maximum(best['mr'], 1e-6), 1.0) ** stepZ, 0.1, 10)
  zf[p] = best['zf']; yStart = best['y']; allWindows += best['windows']; allOnsets.append(best['on'])
  if p == 0: seed0 = best['seed']
  print(f'period {p} done: ZIP err {best["e"]:.3f}', flush=True)

onsets = np.concatenate(allOnsets)
print(f'infected by Apr 20 2020 {onsets[:SERO_DAY].sum()/pop.sum():.1%} | ever infected by Nov 2021 {onsets.sum()/pop.sum():.1%}')
pickle.dump(dict(zf=zf, seed=seed0, windows=allWindows, onsets=onsets, rhoLate=RHO_LATE), open(f'{SCR}/{FIT_FILE}', 'wb'))
