import math

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    output = []
    c = (size-1)//2 

    for i in range(0,size):
        temp = []
        for j in range(0,size):
            temp.append(math.exp(-((i-c)**2+(j-c)**2)/(2*sigma**2)))

        output.append(temp)

    s = sum([sum(i) for i in output])

    return [[i/s for i in a] for a in output]