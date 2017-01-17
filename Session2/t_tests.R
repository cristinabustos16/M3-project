x1 = 86.38
sd1 = 1.04

x2 = 85
sd2 = 1.49
  
n = 5

sxy = (sd1^2 + sd2^2) / 2
t = abs(x1 - x2) / (sxy * sqrt(2 / n))

df = 2 * (n-1)

pval = 2 * pt(t, df, lower.tail = FALSE)

pval