import math
import matplotlib.pyplot as plt

A000041 = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176, 231, 297, 385, 490, 627, 792, 1002, 1255, 1575, 1958, 2436, 3010, 3718, 4565, 5604, 6842, 8349, 10143, 12310, 14883, 17977, 21637, 26015, 31185, 37338, 44583, 53174, 63261, 75175, 89134, 105558, 124754, 147273, 17352]

def hardy_ramnujan_aprox(n):
  if n < 0 : return -1
  if n == 0 : return 1
  return 1 / (4 * n * math.sqrt(3)) * math.exp(math.pi * math.sqrt((2 * n) / 3))

def hardy_ramnujan_aprox_for_all_less_than(n):
  res = []
  for i in range(n + 1):
    res.append(hardy_ramnujan_aprox(i))
  return res

def plot_hardy_ramnujan_aprox_for_all_less_than(n):
  x = list(range(n + 1))
  y = hardy_ramnujan_aprox_for_all_less_than(n)

  plt.plot(x, y, label = "p(n) approx.")
  plt.plot(list(range(len(A000041[:n + 1]))), A000041[:n + 1], label = "p(n)")
  plt.ylabel("p(n)")
  plt.xlabel("n")
  plt.legend()
  plt.show()