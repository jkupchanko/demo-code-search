"""Where the corpus files live.

Split out so the prepare scripts do not have to import anything that touches
Qdrant. They run before there is a cluster to talk to.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
