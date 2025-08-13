from main import main
args = main(['circle_n/config.yaml'])
from model.MOLLM import MOLLM,ConfigLoader

ns = [31,30,29,28,27]
for i in ns:
    config = ConfigLoader(args.config)
    config.config['description'] = f'n = {i} circles in a unit square'
    config.config['save_suffix'] = f'circle_packing_{i}'
    config.config['n_circles'] = i
    mollm = MOLLM(config,resume=args.resume,eval=args.eval,seed=args.seed,objectives=args.objectives,directions=args.directions)
    mollm.run()