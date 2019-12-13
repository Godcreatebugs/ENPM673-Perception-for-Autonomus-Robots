# ENPM673 

#Project 1 - AR Tag Detection


####Running the Code
- To run the code open terminal in the desired trajectory and write the following command:

```
python AR_Tag.py 

```

- This will produce the output video with all the Tag IDs and the 4 corners of the tag detected for the multipleTags video sequence.

There are three command line arguments:

- To run a different video:
```
python AR_Tag.py --video=Tag0
```
- To show lena on the Tags:
```
python AR_Tag.py --video=Tag0 --lena_flag=True
```
- To show the cube on the Tags:
```
python AR_Tag.py --video=Tag0 --cube_flag=True
```

- To run the multipleTags video with lena and cube both copy paste the following command:
```
python AR_Tag.py --video=Tag0 --lena_flag=True --cube_flag=True
```



- Note: To exit the video at any point press "q".
