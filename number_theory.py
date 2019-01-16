
import time
from threading import Semaphore, BoundedSemaphore, Thread
from number_thread import NumberThread
import array
import math, random
import itertools, functools, operator

def absolute(n) :
    if n < 0 :
        return -1 * n
    return n

def gcd(a, b) :
    return math.gcd(b, a % b)

def gcd_iterative(a, b) :
    c = 0
    while b != 0 :
        c = a
        a = b
        b = c % b
    return a

def gcd_tree(a, b) :
    if b == 0 :
        return a
    return gcd(b, a % b)

def modular_mul_inverse(a, n) :
    """Return modular inverse of 'a' w.r.t 'n'
    
    Arguments:
        a {int} -- a
        n {int} -- n (modular)
    """
    if n == 0 :
        return
    if not gcd(a, n) == 1 :
        return
    t, n_t = 0, 1
    r, n_r = n, a
    while n_r != 0:
        quotient = r // n_r
        t, n_t = n_t, t - quotient * n_t
        r, n_r = n_r, r - quotient * n_r
        if r > 1 :
            pass
            # return
        if t < 0 :
            t += n
    return t % n

def pow(a, p, n = 0): 
    """Return a ** n mod(n)  
    
    Arguments:
        a {int} -- base
        p {int} -- exponent
        n {int} -- modular
    """
    result, hare = 1, 2
    while p != 0 :
        if p & 1 == 1 :
            p -= 1
            result = (result * hare) % n if n else result * hare
        hare = (hare * hare) % n if n else hare * hare
        p //= 2
    return result
    # t = int(math.log2(p))
    # k = a ** ((p - 2 ** t)) % n
    # r = a
    # for i in range(t) :
    #     r = r * r % n
    # return r * k % n


def multiply(arr, prime_exp_list = False) :
    """Multiply all element in arr, and return the value
    
    Keyword Arguments:
        arr {list of ints} -- integer or prime exponent list 
        prime_exp_list {bool} -- wether the arr is prime exponent list or not (default: {False})
    """
    # print("Prime list : ",prime_exp_list," arr : ", arr)
    if len(arr) == 1 :
        if prime_exp_list :
            return arr[0][0] ** arr[0][1]
        return arr[0]
    elif len(arr) == 2 :
        if prime_exp_list :
            return arr[0][0] ** arr[0][1] * arr[1][0] ** arr[1][1]
        return arr[0] * arr[1]
    else :
        high = len(arr)
        mid = high // 2;
        return multiply(arr[:mid], prime_exp_list) * multiply(arr[mid:], prime_exp_list)

def mulitiplication_thread_handler(thread, num1, num2, sync_t , res):
    """mulitiplication_thread_handler
    
    Arguments:
        thread {NumberThread} -- thread on which running
        num1 {int} -- number 1
        num2 {int} -- number 2
        sync_t {Semaphore} -- for sync purpose
        res {list of ints} --  result integer array to store result at res[thread.id]
    """
    sync_t.acquire()
    # print("thread_id  : " , thread.id , ", state: " , thread.state , ", num1 : " , num1 , ", num2 : " , num2 , ", sync_t : " , sync_t._value)
    thread.state_in(NumberThread.__WORKING__)
    res[thread.id] = num1 * num2
    # print("result[", thread.id ,"] : ", res[thread.id])
    thread.state_in(NumberThread.__IN_ACTIVE__)
    sync_t.release()

def factorial_consec_prod_linear(low, high) :
    """Mulitplies the consec numbers from low to high exclusive, i.e. numbers in domain [low, high)
    
    Arguments:
        low {int} -- low limit, inclusive
        high {int} --  high limit, exclusive
    
    Returns:
        int -- low * (low + 1) * ... (high - 1)
    """
    result = 1
    for i in range(low, high) :
        result *= i
    return result

def factorial_consec_prod_linear_thread_handler(thread, low, high, sync_t, res):
    """The method used only in case to get factorial_consec_prod on the seprate thread
    
    Arguments:
        low {int} -- low limit, inclusive
        high {int} -- high limit, exclusive
        thread_id {int} -- thread id on which running
        res {list of ints} -- result integer array to store result at res[thread_id]
    """
    sync_t.acquire()
    thread.state_in(NumberThread.__WORKING__)
    res[thread.id] = (factorial_consec_prod_linear(low, high))
    thread.state_in(NumberThread.__IN_ACTIVE__)
    sync_t.release()

def factorial_modular_mul_inverse():
    return

def factorial_linear(n) :
    if n < 2 : return 1
    return factorial_consec_prod_linear(1, n + 1)

def factorial_consec_prod_tree(low, high, m = 0) :
    """Mulitplies the consec numbers from low to high exclusive, i.e. numbers in domain [low, high)
    
    Arguments :
        low {int} -- low limit, inclusive
        high {int} --  high limit, exclusive

    Keyword Arguments :
        m {int} -- modular (default: {0})
    
    Returns:
        int -- low * (low + 1) * ... (high - 1)
    """
    if low > high : 
        if not m :
            return factorial_consec_prod_tree(high, low)
        return factorial_consec_prod_tree(high, low, m) % m
    if low + 1 < high:
        mid = (high + low + 1) // 2
        if not m :
            return factorial_consec_prod_tree(low, mid) * factorial_consec_prod_tree(mid, high)
        return (factorial_consec_prod_tree(low, mid, m) * factorial_consec_prod_tree(mid , high, m)) % m
    if not m :
        return low 
    return low % m


def factorial_consec_prod_tree_thread_handler(thread, low, high, sync_t, res):
    """The method used only in case to get factorial_consec_prod on the seprate thread
    
    Arguments:
        low {int} -- low limit, inclusive
        high {int} -- high limit, exclusive
        thread_id {int} -- thread id on which running
        res {list of ints} -- result integer array to store result at res[thread_id]
    """
    sync_t.acquire()
    # print("typeof : " , type(thread))
    thread.state_in(NumberThread.__WORKING__)
    # print("thread_id  : " , thread.id , ", state: " , thread.state , ", low : " , low , ", high : " , high , ", sync_t : " , sync_t._value)
    # rest = (factorial_consec_prod(low, high))
    res[thread.id] = (factorial_consec_prod_tree(low, high))
    # print("res[", thread.id ,"] : " ,res[thread.id])
    thread.state_in(NumberThread.__IN_ACTIVE__)
    sync_t.release()

def factorial(n) :
    return math.factorial(n)

def factorial_modular(n, m = 0) :
    """Return n! mod (m)
    
    Arguments:
        n {[type]} -- n of n!
    
    Keyword Arguments:
        m {int} -- modular (default: {0})
    """
    
    if n < 2 : return 1
    if m :
        return factorial_consec_prod_tree(1, n + 1, m)
    return factorial_consec_prod_tree(1, n + 1)

def factorial_tree(n) :
    if n < 2 : return 1
    return factorial_consec_prod_tree(1, n + 1)


def factorial_concurrent(num, threads, test) :
    """Finds the factorial of number, using 'threads' numbers of threads concurrently
    
    Arguments:
        num {int} -- integer number to get factorial
        threads {int} -- number of threads to used execute num factorial concurrently
        test {boolean} -- testing purpose
    
    Returns:
        int -- factorial(num) = num! = num * (num - 1) * ... * 3 * 2 * 1
    """
    res = threads * [1] #to stores the result return by thread
    imax = num // threads + 1 #Max interval of multiplication
    i = 0 # Thread iterator
    sync_t = Semaphore(threads)
    threads_arr = []
    low, high = 1, imax
    while low <= num and i < threads :
        high = num + 1 if high > num  else high
        t = NumberThread(id = i, handler= True, target=factorial_consec_prod_tree_thread_handler, args=(low, high, sync_t, res))
        threads_arr.append(t)
        threads_arr[i].start()
        low = high  
        high += imax
        i += 1 

    while sync_t._value <= threads :
        if sync_t._value >= 2 :
            dead_ind = NumberThread.workers_in_state(thread_arr=threads_arr,state = NumberThread.__IN_ACTIVE__)
            if len(dead_ind) >= 2 :
                ind = list(dead_ind.keys())
                num1, num2 = res[dead_ind[ind[0]]], res[dead_ind[ind[1]]]
                res[ind[0]], res[ind[1]] = 0, 0
                threads_arr[ind[0]] = NumberThread(id = dead_ind[ind[0]], handler= True, state= 0, target= mulitiplication_thread_handler, args=(num1, num2, sync_t, res))
                threads_arr[ind[0]].start()
                c = threads_arr.pop(ind[1])
                c = None
            pass

        dead_ind = NumberThread.workers_in_state(thread_arr=threads_arr,state = NumberThread.__IN_ACTIVE__)
     
        if len(threads_arr) == 1 :
            if threads_arr[0].state != NumberThread.__WORKING__ :
                break

    return res[0]
    
def factorial_prime_factorization(n) :
    """factorize the n! return in prime exponents form
    
    Arguments:
        n {int} -- integer n to find factorial
    Returns:
        list of list -- [[p_1, e_1], [p_2, e_3], ... [p_n, e_n]]
    """
    sieve_to_n = sieve_eratosthenes(n)
    result = []
    for p_i in sieve_to_n :
        result.append([p_i, prime_exp_in_factorial(n, p_i)])
    return result

def prime_exp_in_factorial(n, p1) :
    """prime_exp_in_factorial
    
    Arguments:
        n {int} -- factorial term
        p1 {int} -- prime number term
    Returns :
        int -- exponent to p1, in n!
    """
    p1 = factorize(p1, "prime_factors").pop()
    # print("p1 : ", p1)
    if p1 == 1 :
        return n
    t, e, p_t = n, 0, p1
    while p_t <= n :
        t = n // p_t
        e += t
        p_t = p1 * p_t
    return e


def factorial_p(n) :
    arr = factorial_prime_factorization(n)
    # print("list : ", arr)
    return multiply(arr, prime_exp_list= True)

def factorize(n, mode = "factors", result = set()) :
    if mode.lower() == "factors" :
        result = factorize(n, "prime_factors")
        return super_set(result, "multiplication", True, ",", n)
    elif mode.lower() == "prime_factors" :
        if n == 1 : 
            return {1}
        result = set()
        i, mid = 3, math.floor(math.sqrt(n))
        # print("mid : ", mid)
        while n % 2 == 0 :
                result.add(2)
                n = n // 2 
                if is_prime(n) : 
                    return sorted(result.union({n}))        
        while n > 1 :
            # print("result : ", result)
            while n % i == 0 :
                result.add(i)
                n = n // i
                if is_prime(n) :
                    return sorted(result.union({n})) 
            i += 2
            if i > mid :
                if is_prime(n) :
                    return sorted(result.union({n}))
                break
        return sorted(result)

def prime_factor(n, gt = 1, up_limit = 100) :
    """Return prime factor greater than gt_limit
    
    Arguments:
        n {int} -- [description]
    
    Keyword Arguments:
        gt {int} -- greater than limit (default: {1})
        up_limit {int} -- if int then get prime 
    """
    limit =  list(sieve_eratosthenes(up_limit))
    gt = int(math.sqrt(n)) if gt == 1 else gt 
    k, length = 0, len(limit)
    while k < length :
        while n % limit[k] == 0 :
            n //= limit[k]
        if n < gt :
            return n * limit[k]
        k += 1
    return n
         
     

def factorize_tree(n, mode = "factors") :
    if n == 1 :
        return {1}
    if n == 2 :
        return {2}
    result = {1}
    mid , i = math.floor(math.sqrt(n)), 2
    print("n : ", n, " result : ", result)
    while i <= mid :
        if n % i == 0 :
            # print("i : ", i)
            # tset = set()
            # for j in result : 
            #     tset.add(i * j)
            # result = result.union(tset)
            # result = super_set(result, "multiplication", max = n)
            result = result.union(factorize(n // i, mode, result = result))
        i += 1
    return sorted(result.union({n}))

def is_prime(n) :
    return is_prime_trial_div(n)

def is_prime_trial_div(n) :
    if n == 2 :
        return True
    if n & 1 == 0 or n == 1 or n == 0:
        return False
    mid = int(math.sqrt(n)) + 1
    for i in range(3, mid, 2) :
        if n % i == 0 :
            return False
    return True

def is_prime_nCx(n) :
    mid = int(math.sqrt(n))
    if nCx(n + mid, n) % n == 1:
        return True
    return False 


def is_strong_pseudo_fermat_probable_prime(n, limit = 97, depth = 2) :
    """Strong Pseudo Fermat Prime / Miller - Rabin test
    Step 1 : n = 2 ** k * m + 1
    Step 2 : get a = 2 < a < n -1
    Step 3 : Compute b_0 = a ** m mod(n), b_i = (b_i) ** 2
    Arguments:
        n {int} -- integer number to test
    Keywords Arguments : 
        limit {int} -- limit to which perform trial and division
    """
    if depth == 0: 
        return False
    if n == 0 or n == 1:
        return False
    mid = int(math.sqrt(n))
    prime_list = sieve_eratosthenes(limit)
    for i in prime_list :
        if i > mid : 
            return True
        if n % i == 0 :
            return False
    # if n & 1 == 0 or n % 3 == 0 :
    #     return False
    d = 0
    k = n - 1
    # if n == 2 :
    #     return True
    # if n & 1 == 0:
    #     return False
    while k % 2 == 0 :
        d += 1
        k //= 2
    a = random.randint(2, n - 2)
    # a = 2
    m = (n - 1) // (2 ** d)
    while gcd(a, m) != 1 :
        a += 1
    if gcd(m, n) != 1 or gcd(a, n) != 1:
        return False
    b_0 = pow(a, m, n)
    # print("n == ", n, ", b_0 = ", b_0, ", m == ", m)
    if b_0 == 1 or b_0 == n - 1 :
        return True
    start = time.time_ns()
    d += 1
    i = 0
    while i < d:
        b_0 = (b_0 * b_0) % n
        i += 1
        # print("i : ", n, "b_0  :  ", b_0)
        if b_0 == n - 1 :
            return True
        if b_0 == 1 :
            return False
    return is_strong_pseudo_fermat_probable_prime(n = n, depth = depth - 1)

def is_miller_rabin_probable_prime(n) :
    return is_strong_pseudo_fermat_probable_prime(n)

def is_pocklington_probable_prime(n, fermat_little_true = False) :
    if not fermat_little_true :
        if not is_prime_fermat_little_probable_prime(n) :
            return False
    p = prime_factor(n = n - 1) # p is prime number, p > sqrt(n)
    pw = pow(2, (n - 1) // p, n)
    # print("pw : " + pw)
    if gcd(pw, n) == 1 :
        return True
    return False

def is_prime_BPSW(n):
    """BPSW Algorithm
    
    Arguments:
        n {int} -- tester
    """
    if is_miller_rabin_probable_prime(n) :
        if is_pocklington_probable_prime(n) :
            return True
    return False


def is_prime_fermat_little_probable_prime(n, limit = 0) :
    """Fermat Little Probable Prime
    
    Arguments:
        n {int} -- [description]
        limit {int} -- [description]
    """
    if n == 2 :
        return True
    if n & 1 == 0 or n < 2:
        return False
    limit = n - 1 if limit == 0 else limit
    a = 2
    k = 10
    while gcd(a, n -1) != 1 and k > 0 :
        a = random.randint(2, limit)
        k -= 1
    if pow(a, n - 1, n) == 1:
        return True
    # return is_prime_fermat_little_probable_prime(n = n, limit = limit - 1)
    return False

def next_prime(n) :
    """Finds next prime number with complexity of O(n ** 1.5)
    Assumes Riemann Hypothesis to be true...
    if n1 = solve(nlogn(n))
    So next prime number must lie in interval (n1, n + sqrt(n1) * log(n1)) 
    
    Arguments:
        n {int} -- integer number, from which want next prime
    
    Returns:
        int -- next prime number from 'n'
    """
    if n % 2 == 0 :
        return 3
    gap = math.ceil(math.sqrt(n) * math.log(solve_nlogn(n))) + 1
    # print("gap : ", gap)
    # print("cprime : ", n)
    for k in range(n + 1, n + gap, 1):
        counter, mid = 0, math.ceil(math.sqrt(k))
        if k % 2 == 0 or k % 3 == 0:
            continue
        for i in range(3, mid, 2) :
            if k % i == 0 :
                counter = 1
                break
        if counter == 0 :
            return k
    return n

def sieve_eratosthenes(n, sort = False) :
    """Sieve_eratosthenes finds all primes <= n, using sieve_eratosthenes method
    It give processing complexity of O(nlogn)
    And memory complexity of O(n)
    
    Arguments:
        n {int} -- max limit
    
    Returns:
        set -- set of prime numbers <= n
    """
    ifprime = [1] * (n + 1)
    prime_list = set()
    i  = 2
    while i <= n :
        ifprime[i] = 1
        prime_list.add(i)
        # print("ifprime : ", ifprime)
        for j in range(i + i, n + 1, i) :
            ifprime[j] = 0
        i += 1
        try :
            while ifprime[i] == 0 :
                i += 1
        except IndexError :
            pass
    if sort :
        return sorted(prime_list)
    return prime_list

def iterative_primes(n):
    """ iterative primes finds all primes <= n, using next_prime method
    
    Arguments:
        n {int} -- max limit
    
    Returns:
        set -- set of prime numbers <= n
    """
    result = set()
    i = 2
    while i < n :
        # print("result : ", result)
        # print("i : ", i)
        result.add(i)
        i = next_prime(i)
    return result

def pi_x(x) :
    """Prime counting function
    
    Arguments:
        x {int} -- up limit
    
    Returns:
        int -- prime count
    """
    return len(sieve_eratosthenes(x))



def super_set(s_set : set, mode : str = "string", sort = False, delim = ",", max : int = 4294967296):
    """Finds the superset of set, of givien type using the itertools combination method
    
    Arguments:
        set {set} -- given set to find super set
    
    Keyword Arguments:
        mode {str} -- determines kind of superset to return (default: {"string"})
        delim {str} -- only in case of mode == 'string', (default: {","})
        max {int} --act of max value to inculde in math operation
    
    Returns:
        set -- superset of set
    """
    n = len(s_set) + 1
    result = set()
    for i in range(0, n, 1) :
        for s_t in itertools.combinations(s_set, i) :
            if mode == "string" :
                result
            elif mode == "multiplication" :
                try : 
                    # print("s_st : ", s_t)
                    t = functools.reduce(operator.mul, s_t)
                    result.add(t) if max % t == 0 else 1
                except TypeError:
                    pass

    return sorted(result) if sort else result

def solve_nlogn(c, x = 1, precision = 6) :
    """solves the equation nlog(n) == c, to find 'n'
    
    Arguments:
        c {float} -- RHS, float number
        x {float} -- assume solution 
        precision {float} -- assume solution 
    
    Returns:
        float -- solution for above equation
    """
    precision = 10 ** (-precision) if precision > 1 else precision

    x_n1 = x - (x * math.log(x) - c) / (math.log(x) + 1)
    
    if math.fabs(x_n1 - x) > precision :
        return solve_nlogn(c, x_n1, precision)
    else :
        return x_n1



def my_factor(n) :
    print("n : " , n)
    mid = int(math.sqrt(n))
    setn = set()
    prev = 1
    if n == 1 :
        return set()
    if n == 0 :
        return set()
    i = mid
    while i > 1 :
        diff_frac = ((n - i * (n // i)) % i)   
        # print("i : ", i)
        if n % i == 0 :
            # print("true : ", i)
            setn.add(i)
            break
        i -= int(math.sqrt(diff_frac))
    if i == 1 :
        return setn.union({n})
    return setn.union(my_factor(n // i))

def brent_cycle_detectn(x0 , fn) :
    power, lamda = 1, 1
    tortoise = x0
    hare = fn(x0)
    while tortoise != hare :
        if power == lamda :
            tortoise = hare
            power *= 2
            lamda = 0
        hare = fn(hare)
        lamda += 1
    mu = 0
    tortoise , hare = x0, x0
    for i in range(lamda) :
        hare = fn(hare)
    while tortoise != hare :
        tortoise = fn(tortoise)
        hare = fn(hare)
        mu += 1
    return lamda, mu

def floyd(x0, fn) :
    tortoise = fn(x0)
    hare = fn(fn(x0))
    while tortoise != hare :
        tortoise = fn(tortoise)
        hare = fn(fn(hare))

    mu = 0
    tortoise = x0
    while tortoise != hare :
        tortoise = fn(tortoise)
        hare = fn(hare)
        mu += 1
    lamda = 1
    hare = fn(tortoise)
    while tortoise != hare :
        hare = fn(hare)
        lamda += 1
    return lamda, mu

def pollard_rho(n) :
    limit = sieve_eratosthenes(10)
    result = set()
    for val in limit :
        while n % val == 0 :
            n //= val
            result.add(val)
            
    x0 = random.randint(int(math.sqrt(n)), n)
    c = random.randint(1, n)
    tortoise = x0
    hare = psuedo_random_sq_c_fn(x0, c, n)
    cycle_size = 2 #current cycle size
    count = 1
    factor = 1
    while tortoise != hare and factor == 1 :
        if cycle_size == count :
            tortoise = hare
            cycle_size *= 2
            count = 0
        hare = psuedo_random_sq_c_fn(hare, c, n)
        factor = gcd(absolute(tortoise - hare), n) 
        count += 1
    if factor == 1 or factor == n :
        return result.union({n})
    return result.union({n//factor, factor})


def psuedo_random_sq_c_fn(x, c = random.randint(0, 100), mod = 100) :
    return (x * x + c) % mod

def nCx(n, x, m = 0) :
    """n >= x >= 0, i.e. x = [0, n]
    Return nCx, if mod is other than 0 , else returns nCx mod(n)
    
    Arguments:
        n {int} -- n high limit
        x {int} -- x low limit
    
    Keyword Arguments:
        m {int} -- modular (default: {0})
    """
    if m :
        return (factorial_consec_prod_tree(n - x + 1, n + 1, m) * modular_mul_inverse(factorial_modular(x, m), m)) % m
    num = factorial_consec_prod_tree(n - x + 1, n + 1)
    deno = factorial(x)
    return num // deno

