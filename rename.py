import os 
  
# Function to rename multiple files 
def main(): 
	i=38
	for count, filename in enumerate(os.listdir("good_aug")): 
		dst ="bad_"+str(i)+ ".png"
		src ='/home/yuvi/projects/minorproject/openpose/good_aug/'+filename
		dst ='/home/yuvi/projects/minorproject/openpose/good_aug1/'+dst
		i=i+1 
		os.rename(src, dst) 
  
# Driver Code 
if __name__ == '__main__':       
	# Calling main() function 
	main() 