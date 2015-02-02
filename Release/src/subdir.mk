################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Fmincg.cpp \
../src/GradientParameter.cpp \
../src/IOUtils.cpp \
../src/NeuralNetwork.cpp \
../src/NeuralProcessor.cpp 

OBJS += \
./src/Fmincg.o \
./src/GradientParameter.o \
./src/IOUtils.o \
./src/NeuralNetwork.o \
./src/NeuralProcessor.o 

CPP_DEPS += \
./src/Fmincg.d \
./src/GradientParameter.d \
./src/IOUtils.d \
./src/NeuralNetwork.d \
./src/NeuralProcessor.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/ubuntu/AMDAPPSDK-3.0-0-Beta/include/CL -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


