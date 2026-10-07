import sys, pickle
exec(open(sys.argv[1]).read())
tag = f'rho{RHO_LATE}'
saved = pickle.load(open(f'{SCR}/{FIT_FILE}', 'rb')); zf = saved['zf']
rr = tractFactors(zf)
def run(alloc, until=END):
  y = initial(saved['seed'], rr); ons = []
  for ws, we, b in saved['windows']:
    if ws >= until: break
    y, on = step_days(y, ws, min(we, until), b, rr, alloc); ons.append(on)
  return np.concatenate(ons), y
VSTART, MID = 286, 397   # rollout starts Dec 11 2020; mid-rollout snapshot Apr 1 2021
q = pd.qcut(inc, 4, labels=False); QN = ['Poorest', 'Q2', 'Q3', 'Richest']

# ---------- scenarios ----------
scen = {}
for name, alloc in [('No vaccine', None), ('Actual rollout', actualAllocation), ('Uniform', uniformAllocation), ('Income-based', incomeAllocation),
                    ('Susceptible-first', susceptibleAllocation), ('Hotspot', hotspotAllocation(rr))]:
  onsets, _ = run(alloc)
  _, y = run(alloc, MID)
  scen[name] = dict(onsets=onsets, vaxMid=y[1] + y[3] + y[5] + y[7])
pickle.dump(scen, open(f'{SCR}/scenarios2_{tag}.pkl', 'wb'))

print(f'reporting rate after Jun 2020 = {RHO_LATE}; ever infected by Nov 2021 (actual rollout) = {scen["Actual rollout"]["onsets"].sum()/pop.sum():.1%}')
print('Infections Dec 11 2020 - Nov 30 2021:')
for name, s in scen.items():
  inf = s['onsets'][VSTART:].sum(0)
  print(f'  {name:17s} {inf.sum():9.0f} | by income quartile ' + ' '.join(f'{QN[k]} {100*inf[q==k].sum()/pop[q==k].sum():5.1f}%' for k in range(4))
        + ' | ' + ' '.join(f'{BOROS[b]} {100*inf[boro==b].sum()/pop[boro==b].sum():5.1f}%' for b in BOROS))
print('First-dose coverage on Apr 1 2021:')
for name, s in scen.items():
  if name == 'No vaccine': continue
  print(f'  {name:17s} ' + ' '.join(f'{QN[k]} {100*s["vaxMid"][q==k].sum()/pop[q==k].sum():5.1f}%' for k in range(4)))
print('Fitted ZIP transmission factor vs income (pop-weighted mean by income quartile, per period):')
for p in range(len(PERIODS) - 1):
  print(f'  period {p}: ' + ' '.join(f'{QN[k]} {np.average(rr[p][q==k], weights=pop[q==k]):.2f}' for k in range(4)))

