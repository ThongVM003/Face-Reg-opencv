import datetime
import os


# Function to rename multiple files
def main():
    i = 1
    folder = "Datasets/SE171070"
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"{i}.jpg"
        src = f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst = f"{folder}/{dst}"

        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    # main()
    import time
    print(datetime.datetime.utcnow().strftime("%d-%m-%Y-%H:%M:%S"))