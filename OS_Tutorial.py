import os  #make sure you have the os package installed

print(dir(os))
os.getcwd() # prints out the file directory that you are currently in 
os.chdir('c:\eeg\sciencefair22-23') #chdir = change directory; sets the directory to whatever you pass through as a parameter 
os.listdir() # listdir = list directory; returns the names of all the folders/files in this folder 
os.mkdir('directory_name') # creates a single folder 
os.makedirs('directory_name/subDir/subSubDir') #can create a series of layered folders even if the top folder does not yet exist
os.rmdir('directory_name') # remove directory; specifically deletes a certain directory 
os.removedirs('directory_name/subDir/subSubDir') # deletes layers of directories at a time 
os.rename('test.txt', 'demo.txt') # test.txt --|renamed to|--> demo.txt 
os.stat('demo.txt').st_mtime # returns all the statistics on a file like size (.st_size) in bytes, date modified (.st_mtime) 
os.walk() # generates a tuple of three values: directory path, directories within that path, and the files within that path down an entire tree of directories 
for directoryPath, directoryNames, fileNames in os.walk(os.getcwd):
    print('Current Path: ', directoryPath)
    print('Directory Names: ', directoryNames)
    print('File Names : ', fileNames)
os.environ # returns all of your environment variables
os.environ.get('HOME') # returns the path of a specific environment variable

'os.path is a module of os with its own functionalities'
print(dir(os.path))
os.path.join # joins two file paths together with only one "/" in between levels so you don't have to guess every time you work with files 
os.path.exists('file/subfile') # returns a boolean as to whether or not the provided file path exists on the system
os.path.isdir('file/subfile') # returns a boolean as to whether or not the provided argument is a directory ('isfile' to check for files)
os.path.splitext('folder/test.txt') # --> ('folder/test','.txt')







