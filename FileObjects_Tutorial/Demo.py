f = open('test.txt', 'r') # reads a file 
f = open('test.txt', 'w') # writes to a file 
f = open('test.txt', 'r+') # reads AND writes to  a file 
f = open('test.txt', 'a') # appends to a file 

f.name # returns the name of the file 
f.mode # returns the mode that the file is currently in (read, write, or append)

f.close # remember to explicitly close the file when you are done using it

'Using the Context Manager:' 
# context managers allow us to work with file in a specified block of code that closes the file after exiting
with open('test.txt', 'r') as f:
    f_contents = f.read()
    print(f_contents) # prints out the entire file text of 'f' <-- our file

    f_contents = f.readlines() # reads every line of the file
    f_contents = f.readline() # reads the 1st line of the file 
    f_contents = f.readline() # reads the 2nd line of the file 
    print(f_contents, end = '') # end = '' simply gets rid of the line space between each print of the file's lines

    # This method saves more memory because it's not taking in the whole file at once and trying to process it  
    for line in f:
        print(line, end ='')
    f_contents = f.read(100) # first 100 characters of the file (0 to 100)
    f_contents = f.read(100) # next 100 characters of the file (101 to 200)
    f.tell() # returns the character currently read; in this case it would be 200 b/c of the two lines before this one 
    f.seek(0) # sets the reading position to the nth character (so f.seek(0) would set the read to the beginning of the file)

    # prints out the contents of the file while stopping when it reaches the end rather than going over the limit and returning empty chars
    while(f_contents) > 0:
        print(f_contents, end = '')
        f_contents = f.read(200)

# making sure the new file is in write  mode before messing with it 
with open('test2.txt', 'w') as f:
    f.write('test')
    f.seek(0) # makes the next "write" command override the first one's characters
    f.write('Test')

with open('test.txt', 'r') as rf: # read file
    with open('test_copy.txt', 'w') as wf: # write file 
        # for each line in our original file, write that line into "test_copy"
        for line in rf: 
            wf.write(line)











# you can still access information about a file after a context manager has been used for it, you just can read/change info on it
f.closed # returns a boolean as to whether or not the file is closed

