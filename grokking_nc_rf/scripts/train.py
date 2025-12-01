#!/usr/bin/env python

import grok
import os
from datetime import datetime
start_time = datetime.now()

parser = grok.training.add_args()
parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
hparams = parser.parse_args()
hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)


print(hparams)
print(grok.training.train(hparams))
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print("TRAINING COMPLETE")