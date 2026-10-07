# Paper figures: model fit to reported cases, compartments and tract maps for each strategy, and the strategy comparison.
# Plain matplotlib style to match the other figures in the paper. Run after fit.py and scenarios.py.
import sys, pickle
exec(open(sys.argv[1]).read())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = sys.argv[2]
saved = pickle.load(open(f'{SCR}/{FIT_FILE}', 'rb')); rr = tractFactors(saved['zf'])
VSTART, MID = 286, 397   # rollout starts Dec 11 2020; vaccination map on Apr 1 2021
dates = DAY0 + pd.to_timedelta(np.arange(END + 1), 'D')

def runDaily(alloc):
  y = initial(saved['seed'], rr); totals = [y.sum(1)]; yMid = yStart = None
  for ws, we, b in saved['windows']:
    for day in range(ws, we):
      if day == VSTART: yStart = y.copy()
      if day == MID: yMid = y.copy()
      y, _ = step_days(y, day, day + 1, b, rr, alloc); totals.append(y.sum(1))
  return np.array(totals), yStart, yMid, y

geo = geopd.read_file('nycCensusTracts/nyct2020.shp'); geo['GEOID'] = geo['GEOID'].astype(np.int64)
geo = geo[['GEOID', 'geometry']].merge(df[['GEOID']], on='GEOID')   # same tract order as df
assert (geo.GEOID.to_numpy() == df.GEOID.to_numpy()).all()

res = {}
for name, alloc in [('Actual rollout', actualAllocation), ('Uniform', uniformAllocation), ('Income-based', incomeAllocation),
                    ('Susceptible-first', susceptibleAllocation), ('Hotspot', hotspotAllocation(rr))]:
  totals, yStart, yMid, yEnd = runDaily(alloc)
  res[name] = dict(totals=totals, vaxMid=(yMid[1] + yMid[3] + yMid[5] + yMid[7]) / pop,
                   infRollout=(yEnd[8] - yStart[8]) / pop)
  print(f'{name:17s} infections during rollout {(yEnd[8] - yStart[8]).sum():.0f}, recovered at end {(yEnd[6] + yEnd[7]).sum():.0f}, '
        f'vaccinated at end {(yEnd[1] + yEnd[3] + yEnd[5] + yEnd[7]).sum():.0f}')

LABEL = 14

# model fit: reported cases, citywide and by borough
rep = saved['onsets'] * np.array([rho(dd, rhoEarly) for dd in range(END)])[:, None]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dates[:END], citySmooth, label='Observed (7-day average)')
ax.plot(dates[:END], rep.sum(1), label='Model')
ax.set_xlabel('Date', fontsize=LABEL); ax.set_ylabel('Reported Cases per Day', fontsize=LABEL)
ax.grid(True); ax.legend()
fig.tight_layout(); fig.savefig(f'{OUT}/model_fit.png', dpi=150); plt.close(fig)

fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
for ax, b in zip(axes, BOROS):
  n = pop[boro == b].sum() / 1e5
  ax.plot(dates[:END], pd.Series(boroCases[b]).rolling(7, center=True, min_periods=1).mean() / n, label='Observed (7-day average)')
  ax.plot(dates[:END], rep[:, boro == b].sum(1) / n, label='Model')
  ax.set_title(b, fontsize=12); ax.grid(True)
axes[0].legend(); axes[2].set_ylabel('Reported Cases per Day per 100,000', fontsize=LABEL); axes[-1].set_xlabel('Date', fontsize=LABEL)
fig.tight_layout(); fig.savefig(f'{OUT}/model_fit_borough.png', dpi=150); plt.close(fig)

# per-strategy figures first, then the combined ones; all share one y-axis (compartments) and one color scale per map
SLUG = {name: name.lower().replace(' ', '_').replace('-', '_') for name in res}
SERIES = [('Susceptible', lambda T: T[:, 0]), ('Exposed', lambda T: T[:, 2] + T[:, 3]), ('Infected', lambda T: T[:, 4] + T[:, 5]),
          ('Recovered', lambda T: T[:, 6] + T[:, 7]), ('Vaccinated, never infected', lambda T: T[:, 1])]
YMAX = 1.05 * max(f(res[k]['totals'][VSTART:] / 1e6).max() for k in res for _, f in SERIES)   # line plots cover the rollout only
ROWS = [('vaxMid', 'First Dose by April 1, 2021', 'Percentage of Population Vaccinated', 'new_vaccinated_apr2021'),
        ('infRollout', 'Infected Dec 2020 - Nov 2021', 'Percentage of Population Infected', 'new_infected_rollout')]
LIMITS = {key: np.percentile(np.concatenate([100 * res[k][key] for k in res]), [2, 98]) for key, *_ in ROWS}

for name in res:
  T = res[name]['totals'] / 1e6
  fig, ax = plt.subplots(figsize=(10, 6))
  for label, f in SERIES: ax.plot(dates[VSTART:], f(T)[VSTART:], label=label)
  ax.set_xlabel('Date', fontsize=LABEL); ax.set_ylabel('Number of People (millions)', fontsize=LABEL)
  ax.set_xlim(dates[VSTART], dates[-1]); ax.set_ylim(0, YMAX); ax.grid(True); ax.legend()
  fig.tight_layout(); fig.savefig(f'{OUT}/new_seirv_{SLUG[name]}.png', dpi=150); plt.close(fig)
  for key, _, cbar, fname in ROWS:
    g = geo.copy(); g['v'] = 100 * res[name][key]
    fig, ax = plt.subplots(figsize=(8, 7))
    g.plot(column='v', cmap='viridis', vmin=LIMITS[key][0], vmax=LIMITS[key][1], ax=ax, linewidth=0, legend=True, legend_kwds={'label': cbar})
    ax.set_axis_off()
    fig.savefig(f'{OUT}/{fname}_{SLUG[name]}.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# citywide compartments over time, one panel per strategy
fig, axes = plt.subplots(1, len(res), figsize=(22, 5), sharey=True)
for ax, name in zip(axes, res):
  T = res[name]['totals'] / 1e6
  for label, f in SERIES: ax.plot(dates[VSTART:], f(T)[VSTART:], label=label)
  ax.set_title(name, fontsize=LABEL); ax.set_xlim(dates[VSTART], dates[-1]); ax.set_ylim(0, YMAX); ax.grid(True)
  ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
  ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b\n%Y'))
axes[0].set_ylabel('Number of People (millions)', fontsize=LABEL)
fig.supxlabel('Date', fontsize=LABEL)
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(); fig.savefig(f'{OUT}/seirv_strategies.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# tract maps: % with a first dose on Apr 1 2021 (top), % infected during the rollout (bottom); one color scale per row
fig, axes = plt.subplots(2, len(res), figsize=(22, 9))
for r, (key, rowTitle, cbar, *_) in enumerate(ROWS):
  vmin, vmax = LIMITS[key]
  for ax, name in zip(axes[r], res):
    g = geo.copy(); g['v'] = 100 * res[name][key]
    g.plot(column='v', cmap='viridis', vmin=vmin, vmax=vmax, ax=ax, linewidth=0)
    ax.set_axis_off(); ax.set_title(name if r == 0 else '', fontsize=LABEL)
  axes[r, 0].text(-0.05, 0.5, rowTitle, transform=axes[r, 0].transAxes, rotation=90, ha='right', va='center', fontsize=LABEL)
  fig.colorbar(plt.cm.ScalarMappable(matplotlib.colors.Normalize(vmin, vmax), 'viridis'), ax=axes[r], fraction=0.015, pad=0.01, label=cbar)
fig.savefig(f'{OUT}/maps_strategies.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# strategy comparison: share of each income quartile infected during the rollout
scen = pickle.load(open(f'{SCR}/scenarios2_rho{RHO_LATE}.pkl', 'rb'))
q = pd.qcut(inc, 4, labels=False); QN = ['Poorest', 'Q2', 'Q3', 'Richest']
names = [k for k in scen if k != 'No vaccine']
fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(len(names)); w = 0.8 / 4
infQ = np.array([[100 * scen[name]['onsets'][VSTART:].sum(0)[q == j].sum() / pop[q == j].sum() for j in range(4)] for name in names])
for j in range(4):
  ax.bar(x + (j - 1.5) * w, infQ[:, j], w, label=QN[j], color=plt.cm.Blues(0.35 + 0.2 * j))
ax.set_xticks(x, names)
ax.set_xlabel('Allocation Strategy', fontsize=LABEL); ax.set_ylabel('Percentage Infected', fontsize=LABEL)
ax.set_ylim(0, 25); ax.grid(True, axis='y'); ax.set_axisbelow(True)
ax.legend(title='Census Tract Income Quartile', ncol=4, loc='upper center')
fig.tight_layout(); fig.savefig(f'{OUT}/strategies_by_quartile.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5)); d = dates[1:]
for name in names: ax.plot(d[VSTART - 30:], scen[name]['onsets'][VSTART - 30:].sum(1), label=name)
ax.set_xlabel('Date', fontsize=LABEL); ax.set_ylabel('New Infections per Day', fontsize=LABEL)
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b\n%Y'))
ax.grid(True); ax.legend()
fig.tight_layout(); fig.savefig(f'{OUT}/strategies_over_time.png', dpi=150); plt.close(fig)
