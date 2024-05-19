import numpy as np

# x = [3, 5, 7 , 8 , 9, 3]

x = np.zeros((5, 5))

k1 = np.random.randint(0, 5)
k2 = np.random.randint(0, 5)
#k1 = 2
#k2 = 3
k12 = (k1, k2)  

x[k12] = 1
i = 0
for i in range(5):
    response = input('guess where the ship is? input as "x y";')
    response = response.split()

    print(response)

    userx = int(response[0])
    usery = int(response[1])

    if userx >= 5 or usery >= 5:
        print('out of bounds')
        print(x)
        exit()  
    elif k1 == usery and k2 == userx:
        print('hit')
    else:
        print('miss')

print(x)
