#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import pylab

fig = pylab.figure()
figlegend = pylab.figure(figsize=(3, 2))
ax = fig.add_subplot(111)
lines = ax.plot(
    range(10),
    pylab.randn(10),
    "k--",
    range(10),
    pylab.randn(10),
    "k-",
)
figlegend.legend(lines, ("Linear Interference", "No Interference"), "center")
fig.show()
figlegend.show()
figlegend.savefig("legend_interference.png", bbox_inches="tight", dpi=400)
