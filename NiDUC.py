from PIL import Image
import numpy
import imageio
import matplotlib.pyplot as plt
import reedsolo
import random

G = numpy.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 1],        # codeword generator matrix
                 [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0]], numpy.uint8)
G = G.transpose()

H = numpy.array([               # parity-check matrix
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0]], numpy.uint8)

R = numpy.array([               # decoding matrix
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0]], numpy.uint8)

def openImg(file_name):
    image = Image.open(file_name)
    image.load()
    image = image.convert("L")                                          # converting to greyscale
    image = numpy.array(image)                                          # casting converted PIL.Image to numpy.array
    print('data type: ' + str(image.dtype))
    print('array size: ' + str(image.shape))
    return image

def applyBSC(arr, Pb):
    arr = numpy.array(arr)
    for i in range(arr.size):
        noise = 0
        for x in range(8):
            noise *= 2                                                  # shifting left
            if random.random() < Pb:
                noise += 1                                              # flipping
        arr.itemset(i, arr.item(i)^noise)
    return arr

def applyGilbert(arr, Pd, Pz, Pdz, Pzd):
    arr = numpy.array(arr)
    D = True
    for i in range(arr.size):
        noise = 0
        for x in range(8):
            noise *= 2
            if D:
                if random.random() < Pdz:
                    D = False                                           # switching to state Z
                if random.random() < Pd:
                    noise += 1
            else:
                if random.random() < Pzd:
                    D = True                                            # switching to state D
                if random.random() < Pz:
                    noise += 1
        arr.itemset(i, arr.item(i)^noise)
    return arr

def encodeTMR(arr):
    arr_out = numpy.empty(arr.size*3, numpy.uint8)
    for i in range(arr.size):
        arr_out.itemset(3*i, arr.item(i))
        arr_out.itemset(3*i+1, arr.item(i))
        arr_out.itemset(3*i+2, arr.item(i))
    return arr_out

def decodeTMR(arr, shape):
    arr_out = numpy.empty(shape, numpy.uint8)
    for i in range(arr_out.size):
        arr_out.itemset(i, (arr.item(3*i)&(arr.item(3*i+1)|arr.item(3*i+2)))|(arr.item(3*i+1)&arr.item(3*i+2)))      # picking majority
    return arr_out

def encodeHamming(arr):
    global G
    arr_out = numpy.empty(arr.size*2, numpy.uint8)
    datawordArr = numpy.empty(4, numpy.uint8)
    for i in range(arr.size):
        dataword = arr.item(i)
        for h in range(2):                                              # halfword (4 data bits)
            for x in range(4):
                datawordArr.itemset(x, dataword%2)
                dataword //= 2                                          # integer division
            codewordArr = datawordArr.dot(G)
            codeword = 0
            for x in range(8):
                codeword *= 2
                codeword += (codewordArr.item(7-x)%2)
            arr_out.itemset(2*i+h, codeword)
    return arr_out

def decodeHamming(arr, shape):
    global H
    global R
    arr_out = numpy.empty(shape, numpy.uint8)
    codewordArr = numpy.empty(8, numpy.uint8)
    detected = 0
    for i in range(arr_out.size):
        dataword = 0
        for h in range(2):
            codeword = arr.item(2*i+1-h)
            for x in range(8):
                codewordArr.itemset(x, codeword%2)
                codeword //= 2
            errorcodeArr = H.dot(codewordArr)
            errorcode = 0
            for x in range(4):
                errorcode *= 2
                errorcode += (errorcodeArr.item(x)%2)                                   # convert errorcode into a number
            if errorcode > 0:                                                           # if errorcode != 0, error detected
                if errorcode >= 8:                                                      # if first bit of 4bit errorcode is 1, error correctable
                    if errorcode > 8:                                                   # if errorcode = 1000 correct bit 8(index 7),
                        errorcode -= 8                                                  # else correct bit XXX where XXX are 3 least sagnificant bits of errorcode
                    codewordArr.itemset(errorcode-1, codewordArr.item(errorcode-1)^1)   # fliping corrected bit
                else:
                    detected += 1
            datawordArr = R.dot(codewordArr)
            for x in range(4):
                dataword *= 2
                dataword += (datawordArr.item(3-x)%2)
        arr_out.itemset(i, dataword)
    print('detected uncorrectable: ' + str(detected) + '(' + str(100*detected/arr.size) +'%)')
    return arr_out

def interleave(arr):
    arr_out = numpy.empty(arr.size, numpy.uint8)
    sizeMod = arr.size%8
    for i in range(0, arr.size - sizeMod, 8):               # looping through 8byte chunks in arr
        for j in range(7, -1, -1):                          # looping through bytes in chunk
            dataword = arr.item(i+j)
            for k in range(8):                              # looping through bits in byte
                arr_out.itemset(i+k, arr_out.item(i+k)*2)
                arr_out.itemset(i+k, arr_out.item(i+k)+dataword%2)
                dataword //= 2
    for i in range(arr.size - sizeMod, arr.size):           # adding remaining bytes
        arr_out.itemset(i, arr.item(i))
    return arr_out
        
def encodeRS(arr):
    arrShape = arr.shape            # storing shape
    arr.resize(arr.size)
    rs = reedsolo.RSCodec(4)
    sizeMod = arr.size%4
    arr_out = numpy.empty(arr.size*2 - sizeMod, numpy.uint8)
    for i in range(0, arr.size - sizeMod, 4):               # looping through 4byte chunks in arr
        arr_out[2*i:2*i+8] = numpy.frombuffer(rs.encode(arr[i:i+4].tobytes()), numpy.uint8)     #inserting the encoded bytes into the output array
    for i in range(sizeMod):
        arr_out.itemset((arr.size-sizeMod)*2+i, arr.item(arr.size-sizeMod+i))
    arr.resize(arrShape)
    return arr_out

def decodeRS(arr, shape):
    rs = reedsolo.RSCodec(4)
    sizeMod = arr.size%4
    arr_out = numpy.empty(shape, numpy.uint8)
    arr_out.resize(arr_out.size)
    for i in range(0, arr_out.size - sizeMod, 4):               # looping through 8byte chunks in arr
        try:
            arr_out[i:i+4] = numpy.frombuffer(rs.decode(arr[2*i:2*i+8].tobytes()), numpy.uint8)     #inserting the decoded bytes into the output array
        except reedsolo.ReedSolomonError:
            arr_out[i:i+4] = arr[2*i:2*i+4]
    for i in range(sizeMod):
        arr_out.iteset(arr_out.size-sizeMod+i, arr.item((arr_out.size-sizeMod)*2+i))
    arr_out.resize(shape)
    return arr_out


def BER(arrA, arrB):
    count = 0
    for i in range(arrA.size):
        err = arrA.item(i)^arrB.item(i)                                     # XOR finds error bits
        for x in range(8):
            count += err%2                                                  # counting errors
            err //= 2
    return count/(arrA.size*8)

#Start
random.seed(100)
file_name = 'image.png'
myImg = openImg(file_name)


#BSC
print('BSC')
Pb = 0.01                                                           # probabilty of error(flipping a bit)
print('Pb = ' + str(Pb))
myImgBSC = applyBSC(myImg, Pb)
print('BER: ' + str(BER(myImgBSC, myImg)))


#Gilbert
print('Gilbert')
Pd = 0.0                                                              # probabilty of error in D (good) state
Pz = 0.000001                                                            # probabilty of error in Z (bad) state
Pdz = 0.001                                                       # probabilty of switching to state Z
Pzd = 0.1                                                       # probabilty of switching to state D
print('Pd = ' + str(Pd) + ', Pz = ' + str(Pz) + ', Pdz = ' + str(Pdz) + ', Pzd = ' + str(Pzd))
myImgGilbert = applyGilbert(myImg, Pd, Pz, Pdz, Pzd)
print('BER: ' + str(BER(myImgGilbert, myImg)))


#TMR
print('TMR+BSC')
myImgTMR = encodeTMR(myImg)
myImgTMRBSC = applyBSC(myImgTMR, Pb)
myImgTMRBSCdec = decodeTMR(myImgTMRBSC, myImg.shape)
print('BER: ' + str(BER(myImgTMRBSCdec, myImg)))


print('TMR+Gilbert')
myImgTMRGilbert = applyGilbert(myImgTMR, Pd, Pz, Pdz, Pzd)
myImgTMRGilbertdec = decodeTMR(myImgTMRGilbert, myImg.shape)
print('BER: ' + str(BER(myImgTMRGilbertdec, myImg)))

#Hamming(8,4)
print('Hamming+BSC')
myImgHamming = encodeHamming(myImg)
myImgHammingBSC = applyBSC(myImgHamming, Pb)
myImgHammingBSCdec = decodeHamming(myImgHammingBSC, myImg.shape)
print('BER: ' + str(BER(myImgHammingBSCdec, myImg)))

print('Hamming+Gilbert')
myImgHammingGilbert = applyGilbert(myImgHamming, Pd, Pz, Pdz, Pzd)
myImgHammingGilbertdec = decodeHamming(myImgHammingGilbert, myImg.shape)
print('BER: ' + str(BER(myImgHammingGilbertdec, myImg)))

print('Hamming+Interleaving+Gilbert')
myImgHammingInterleaved = interleave(myImgHamming)
myImgHammingInterleavedGilbert = applyGilbert(myImgHammingInterleaved, Pd, Pz, Pdz, Pzd)
myImgHammingInterleavedGilbertInterleaved = interleave(myImgHammingInterleavedGilbert)
myImgHammingInterleavedGilbertInterleaveddec = decodeHamming(myImgHammingInterleavedGilbertInterleaved, myImg.shape)
print('BER: ' + str(BER(myImgHammingInterleavedGilbertInterleaveddec, myImg)))

#ReedSolomon
print('ReedSolomon+BSC')
myImgRS = encodeRS(myImg)
myImgRSBSC = applyBSC(myImgRS, Pb)
myImgRSBSCdec = decodeRS(myImgRSBSC, myImg.shape)
print('BER: ' + str(BER(myImgRSBSCdec, myImg)))

print('ReedSolomon+Gilbert')
myImgRS = encodeRS(myImg)
myImgRSGilbert = applyGilbert(myImgRS, Pd, Pz, Pdz, Pzd)
myImgRSGilbertdec = decodeRS(myImgRSGilbert, myImg.shape)
print('BER: ' + str(BER(myImgRSGilbertdec, myImg)))
