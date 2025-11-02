__name__ = "forecopy"
__version__ = "0.1.0.9000"

from forecopy.cov import cscov, tecov
from forecopy.tools import cstools, tetools
from forecopy.lsrec import csrec, terec

__all__ = ["cscov", "tecov", "cstools", "tetools", "csrec", "terec"]
