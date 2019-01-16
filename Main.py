import number_theory as NT
import time
import math, random
import traceback
from decimal import *
# import plotly
# import plotly.graph_objs as go 
contex = Context(prec=48)
setcontext(contex)
# plotly.tools.set_credentials_file(username="rushike.ab1", api_key="4294967296ni")
# n = 1000
# n = 15008970004754675444678754445699
# n = 3892347
# n = 43490343434
n = 29400438337882878
# n = 4900073056313813

test = set()
for k in range(100) :
    test.add(random.randint(6, 100000000000000))
# print("test array : ", test)
start = time.time_ns()
# print("factors of : ", n ," are ", NT.pollard_rho(n) )
count = 0
for num in test :
    if NT.is_prime(num) :
       count += 1
print("prime is : ", count) 
print("______________________________\n")
start2 = time.time_ns()
count = 0
for num in test :
    # print("factor : ", NT.pollard_rho(num))
    if NT.is_strong_pseudo_fermat_probable_prime(num) :
       count += 1
print("prime is : ", count) 
# print("factors of : ", n, " are ",NT.factorize(n, "prime_factors") )
start3 = time.time_ns()
print("pow : ", NT.pow(2, 1000, 1000))
start4 = time.time_ns()
for num in test :
    print("factor of : ", num," are ", NT.pollard_rho(num))
start5 = time.time_ns()
print("\n\nTime  : " + str((start2 - start) // 1000000) + ", Time  : " + str((start3 - start2) // 1000000), ",\nTime  : " + str((start4 - start3) // 1000000), ", Time  : " + str((start5 - start4) // 1000000))

# # solv = NT.solve_nlogn((start2 - start))
# solv = n * math.log(n, math.e)
# es = (start3 - start2) * (start2 - start) / solv
# try :
#     reas = math.log10(es) / math.log10(n)
     
# except ValueError as e:
#     reas = 0
#     traceback.print_exc()
# except ZeroDivisionError as e:
#     reas = 0
#     traceback.print_exc()
# print("sieve : ", (start2 - start), ",\niterative : ", (start3 - start2),  ",\nsolv x : ", solv, ",\nraised to : ", reas)
