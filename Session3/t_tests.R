x1 = 86.3371
sd1 = 3.6781

x2 = 85.0079
sd2 = 2.8188
  
n = 5

sxy = (sd1^2 + sd2^2) / 2
t = abs(x1 - x2) / (sxy * sqrt(2 / n))

df = 2 * (n-1)

pval = 2 * pt(t, df, lower.tail = FALSE)

pval