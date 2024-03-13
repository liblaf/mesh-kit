import pathlib
import re

import matplotlib.pyplot as plt

log_file = pathlib.Path("run.log")
log_text: str = log_file.read_text()
lines: list[str] = log_text.splitlines()
residuals: list[float] = []
for line in lines:
    match = re.search(r"residual = ([\d\.]+)", line)
    if match:
        residuals.append(float(match.group(1)))
plt.figure()
plt.plot(residuals)
plt.xlabel("Iter")
plt.ylabel("Residual")
plt.yscale("log")
plt.show()
