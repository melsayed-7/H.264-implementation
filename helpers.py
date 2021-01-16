import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def fixdims(img,frows,fcols): # fix the dimensions to make them multiples of the frame size
  rows,cols=img.shape #get the original rows and colomns and discard the cannal dimension
  rows-=rows%frows
  cols-=cols%fcols
  img=img[:rows,:cols]
  return img

def getbasis(u,v,brows,bcols):#helper function to get the basis that will be used in DCT
  basis=np.zeros((brows,bcols))
  for i in range(brows):
    for j in range(bcols):
      basis[i,j]=np.cos((2*i+1)*u*np.pi/16)*np.cos((2*j+1)*v*np.pi/16)
  return basis

def DCT(frame):
  frows,fcols=frame.shape
  DCTmat=np.zeros((frows,fcols))
  for u in range(frows):
    for v in range(fcols):
      basis=getbasis(u,v,frows,fcols)
      DCTmat[u,v]=np.sum(np.multiply(basis,frame))#the correlation operation
  #scaling operations to remove energy at the first rows
  DCTmat[0,:]/=32
  DCTmat[:,0]/=32
  DCTmat[0,0]*=16
  DCTmat[1:,1:]/=16
  return DCTmat

def IDCT(DCTmat):#the function multipy each basis with the corresponding scale and adds them up
  frows,fcols=DCTmat.shape
  frame=np.zeros((frows,fcols))
  for u in range(frows):
    for v in range(fcols):
      basis=getbasis(u,v,frows,fcols)
      frame+=DCTmat[u,v]*basis
  return frame

def quantize(DCTmat,Q):
  DCTmat=np.divide(DCTmat,Q)
  DCTmat=np.round(DCTmat)

def dequantize(DCTmat,Q):
  DCTmat=np.multiply(DCTmat,Q)

def error(oimg,nimg):
  return np.sum(np.square(np.subtract(oimg,nimg)))

#########################################################################
#zigzag transformation

def takestwoD(arr):
  arr = np.asarray(arr);
  m , n = arr.shape
  result = [0 for i in range(n*m)]
  result = np.asarray(result);
  
  
  count = 0;
  for i in range(0,n + m):
    if(i%2 == 0):
      x = 0;
      y = 0;
      if (i<m):
        x = i;
      else:
        x = m - 1;
      if (i<m):
        y = 0;
      else:
        y = i - m + 1;
      while (x >= 0 and y < n):
        result[count] = arr[x][y];
        count = count +1;
        x = x - 1;
        y = y + 1;
    else:
      x = 0;
      y = 0;
      if (i<n):
        x = 0;
      else:
        x = i - n + 1;
      if (i<n):
        y = i;
      else:
        y = n - 1;
      while (x < m and y >= 0):
        result[count] = arr[x][y];
        count = count +1;
        x = x + 1;
        y = y - 1;
  return np.asarray(result);

#########################################################################
#inverse zigzag transformation
#Generic Code
#1D to 2D
def takesoneD(arr,rows,cols):
  m = rows;
  n = cols;
  result = [[0 for i in range(n)] for j in range(m)] 
  result = np.asarray(result)
  arr = np.asarray(arr);
  
  
  count = 0;
  for i in range(0,n + m):
    if(i%2 == 0):
      x = 0;
      y = 0;
      if (i<m):
        x = i;
      else:
        x = m - 1;
      if (i<m):
        y = 0;
      else:
        y = i - m + 1;
      while (x >= 0 and y < n):
        result[x][y] = arr[count];
        count = count +1;
        x = x - 1;
        y = y + 1;
    else:
      x = 0;
      y = 0;
      if (i<n):
        x = 0;
      else:
        x = i - n + 1;
      if (i<n):
        y = i;
      else:
        y = n - 1;
      while (x < m and y >= 0):
        result[x][y] = arr[count];
        count = count +1;
        x = x + 1;
        y = y - 1;
  return np.asarray(result);

#########################################################################
## run length
def run_length(st):

    #prob_0 = count_0 / len(st)
    #prob_1 = count_1 / len(st)

    
    #print(prob_0//prob_1)

    st = list(st)
    list_1 = []
    encoded = []

    for i in range(len(st)):
        
        if(st[i]!=0):
    
            length = len(list_1)
            if(length > 0):
                list_1 = []
                encoded.append(0)
                encoded.append(length)
                encoded.append(st[i])
            else:
                encoded.append(st[i])
        else:
            list_1.append(0)
            
    length = len(list_1)
    if(length > 0):
        list_1 = []
        encoded.append(0)
        encoded.append(length)
    encoded = np.asarray(encoded)
    return encoded

########################################
## reverse run_length

def reverse_run_length(encoded):
    encoded  = list(encoded)
    decoded = []
    
    flag= 0
    for i in range(len(encoded)):
      
      if (flag ==0):
        
        if(encoded[i]!=0):
          flag =0
          decoded.append(encoded[i])

        elif(encoded[i]==0):
          for j in range(encoded[i+1]):
            decoded.append(0)
          flag = 1
          continue
      
      flag = 0
          
    return decoded


########################################
## Huffman encoding

import heapq
import os

class node:
    def __init__(self, symbol, frequency):
        self.symbol = symbol
        self.freq = frequency
        self.left = None
        self.right = None


    def __lt__(self, other):
        if(other == None):
            return -1
        if(not isinstance(other, node)):
            return -1

        return self.freq > other.freq



class Huffman_encoding:
    def __init__(self):
        self.heap = []    # for sorting 
        self.codes = {}  # the mapping from symbol to code
        self.reverse_mapping = {} # the mapping from code to symbol



    # this counts frequency of each symbol
    def make_freq_dict(self, message):
        frequency = {}
        for symbol in message:
            if not symbol in frequency:
                frequency[symbol] = 0
          
            frequency[symbol] += 1
        
        return frequency


    # builds the heap 
    def build_heap(self, frequency):
        for key in frequency:
            n = node(key, -frequency[key])
            heapq.heappush(self.heap, n)

    # merge the nodes of the least 2 probablistic symbols
    def merge_nodes(self): 
        while(len(self.heap)> 1):
            node_1 = heapq.heappop(self.heap)
            node_2 = heapq.heappop(self.heap)

            merged_node = node(None, node_1.freq + node_2.freq)
            merged_node.left = node_1
            merged_node.right = node_2

            heapq.heappush(self.heap, merged_node)


    # builds the code of each symbol
    def helper_function(self, root, code):
        if (root == None):
            return

        if (root.symbol != None):
            self.codes[root.symbol] = code
            self.reverse_mapping[code] = root.symbol
            return
        
        self.helper_function(root.left, code + "0")
        self.helper_function(root.right, code + "1")

    # getting the code
    def make_codes(self):
        root = heapq.heappop(self.heap)
        code = ""
        self.helper_function(root, code)



    # getting the encoded message
    def get_encoded_message(self, message):
        encoded_message = ""
        for m in message:
            encoded_message += self.codes[m]

        return encoded_message


    # wrapper of all huffman functions
    def compress(self, message):
        

        freq = self.make_freq_dict(message=message)
        self.build_heap(freq)
        self.merge_nodes()
        self.make_codes()
        encoded_message = self.get_encoded_message(message)
        
        

        print("Compressed")
        return encoded_message    


    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = []

        for bit in encoded_text:
            current_code += bit
            if(current_code in self.reverse_mapping):
                character = self.reverse_mapping[current_code]
                decoded_text.append(character)
                current_code = ""

        return decoded_text
      ###################################################################3
      #Encoding function
def encode(gray,frows,fcols,Q):
    #fix image dimensions to make it multiple of the frame rows and columns
    rows,cols=gray.shape    
    gray=fixdims(gray,frows,fcols) 
    
    #get the original size of the ppicture by multiplying its number of elemnts * the size of each item in bits 
    osize=gray.size * gray.itemsize 


    finalvec=[] # this is the vector that should hold all the runlength of every frame
    for r in range(int(rows/frows)):
        for c in range(int(cols/fcols)):
            frame=gray[r*frows:(r+1)*frows,c*fcols:(c+1)*fcols]#crop every frame in the picture according to the frame size and the indecies of the loop
            DCTmat=DCT(frame) #it turned out that the frame of size 8 gets the least error when implementing DCT,the 4 and 16 stil get a relatively low error 
            quantize(DCTmat,Q)
            DCTmat1D = takestwoD(DCTmat) #transform the DCT matrix into 1D to perfom runclength code
            encoded = run_length (DCTmat1D) # perform run-length code
            finalvec.extend(encoded) # concatenate every frame with the previous ones

    finalvec=np.asarray(finalvec)
    huffman = Huffman_encoding()
    encoded_img = huffman.compress(finalvec)  #encoded message as a sting of zeros and ones
    csize=len(encoded_img) #get the size (number of bits) of the generated code
    print("compression efficiency is ",csize/osize)
    return encoded_img,huffman # it passes the encoded image and the huffman object
###################################################################3
#Decoding function
def decode(encoded_img,huffman,rows,cols,frows,fcols,Q):
    recimage=np.ones((rows,cols))# intialize recovered image
    decoded = huffman.decode_text(encoded_img)
    decoded=reverse_run_length(decoded)# expand the runlength code

    for r in range(int(rows/frows)):
        for c in range(int(cols/fcols)):
            DCTmat1D=decoded[0:frows*fcols]#get the all of the frame elements and pop it from the list
            del decoded[0:frows*fcols]
            DCTmat=takesoneD(DCTmat1D,frows,fcols) #convert it to 2D array again
            dequantize(DCTmat,Q)
            frame=IDCT(np.asarray(DCTmat))
            recimage[r*frows:(r+1)*frows,c*fcols:(c+1)*fcols]=frame #put each frame in its proper place
    return recimage























