	NeuralNetworkOpenCL is developed as subproject of u-db.org (public EEG database/EEG silent speech detection project) which is a prediction application using machine learning alghoritms(forward/back propagation and gradient descent). While I was studying Andrew NG coursera machine learning course I have developed C/C++ version according to tutorials.During my study I have also implemented Fmincg octave source into C language which you can find in the src folder. As I am originally a java developer NeuralNetwork is my one of first C/C++ application so I advise you to use it with caution. Anyway I would like to inform you that I was able to use application in many cases without any problem (including Andrew NG examples or EEG data analysis) Application has OpenCL also you can save your existing  results and continue from this results in your next training. To run application you need to install AMD OpenCL and export $AMDAPPSDKROOT as root folder of your sdk.


Parameters:

--help	This help info

-x	X(input) file path (just space delimiter accepted currently)

-y	Y(expected result) file path

-r	Rowcount of X or Y file (should be equal)

-c	Column count of X file (each row should have same count)

-n	Number of labels in Y file (how many expected result)

-t	Total layer count for neural network(including X)

-h	Hidden layer size (excluding bias unit)

-i	Number of iteration for training

-l	Lambda value

-p	Do prediction for each input after training complete (0 for disable, 1 for enable, default 1)

-tp	Theta path. If you have previously saved a prediction result you can 	continue from this result by loading from file path. (-lt value should be 	set to 1)

-lt	Load previously saved thetas (prediction result)
	(0 for disable 1 for enable default 0) (-tp needs to be set)

-st	Save thetas (prediction result)(0 for disable 1 for enable default 1)


Please see http://www.u-db.org for more details

 
Installation:
 Application using Posix threads and OpenCL so since you installed OpenCL any linux machine should run without any problem. I just tested on ubuntu so please let me know if you are successful on mac,redhat,suse or any other distro. (burak@linux.com)

After you download the source code, locate into Release folder:

cd NeuralNetworkOpenCL/Release
make clean
make all

If your build successfull then you will see finished message, if not please inform me I will try to help.

How does it work:
 It loads all inputs into memory then share rows per GPU core. So every single shader core will do a neural calculation. According to private memory limits application doesnt support more than 1000 layer item. (i.e. 400 column, 10 label, 3 hiddenlayer with the size of 25 will not fail but 1000 column... will fail.) Let me know if you have god damn fast gpu then I will send you updated source according your needs or update this source code.


Example Usage:
 application comes with sample input data. Again samples based on Andrew NG tutorial which contains various pixel data of numbers from 1 to 10 for image recognition. Since you train with more iterations you will notice that prediction rate will increase. If you have less then 10000 row I recommend you to use multicore version.

 Running the samples (x.dat and y.dat file in release folder) you can copy below code:

 ./NeuralNetworkOpenCL -x x.dat -y y.dat -r 5000 -c 400 -n 10 -t 3 -h 25 -i 10 -l 1 -p 1 -st 0

To increase the iteration and learning rate simply increase -i parameter. And if you want to save results set -st parameter to 1. When training finish you will see an output mentioning “thetas_xxxx.dat file has been saved.” So for your next trainign copy “thetas_xxxx.dat” then set -lt to 1 and -tp to “thetas_xxxx.dat”. Then your params should look similar to below:

./NeuralNetworkOpenCL -x x.dat -y y.dat -r 5000 -c 400 -n 10 -t 3 -h 25 -i 10 -l 1 -p 1 -st 1 -lt 1 -tp  thetas_xxxx.dat

For other params you can run --help.


Performance:
 I have amd 8350 8 core processor overlocked to 4.6 ghz and 16 gb ram, sapphire toxic r9 280x. On my system each neural calculation took:
for 8 core: 20 ms.

Memory:
 NeuralNetwork designed to handle EEG data so quite memory hungry. All inputs will be loaded into memory including hiddenlayer, error and delta values. Valgrind (memory leak tool for C++) reports no error and on my system it is using aroung 1gb memory for the sample data. I have also developed Cuda and MPI versions for large dataset to share every single iteration between machines. Soon I will be sharing them too.

Exception handling:
 I have left my job to complete this study and I learn a lot. But unfortunately I dont have more time/resources to include improved exception handling,comments. So again: use with caution!

License:
 All code under BSD 2 license if you have conflict let me know I will do my best to help.
