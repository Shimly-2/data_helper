from multiprocessing import Pool

def multiprocessing_helper(fn, args_list, pool_num=None):
    ret = []
    if pool_num is None:
        pool = Pool()
    else:
        pool = Pool(processes=pool_num)
    results = []
    for args in args_list:
        results.append(pool.apply_async(fn, args=args))
    pool.close()
    pool.join()
    
    for res in results:
        ret.append(res.get())
    
    return ret