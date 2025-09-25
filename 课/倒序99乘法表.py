
for i in range (-9,0):
    for j in range (1,-i+1):
        print (f"{-i}*{j}={-i*j}", end = "\t")
    print()


#更好的方法
for i in range (9,0,-1):
    for j in range (i,0,-1):
        print (f"{i}*{j}={i*j}", end = "\t")
    print()    #print()即为print("\n")，默认的换行符
    
