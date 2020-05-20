#include "mbed.h"
#include <cmath>
#include "DA7212.h"

#include "uLCD_4DGL.h"

#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define bufferLength (32)

DA7212 audio;
int16_t waveform[kAudioTxBufferSize];

DigitalOut led1(LED1);
DigitalOut led2(LED2);
DigitalOut led3(LED3);

InterruptIn sw2(SW2);
InterruptIn sw3(SW3);

Serial pc(USBTX, USBRX);
uLCD_4DGL uLCD(D1, D0, D2);

EventQueue queueDNN(32 * EVENTS_EVENT_SIZE);

Thread threadDNN(osPriorityNormal, 120*1024);

int music = 0;
int mode = 0;
int selection = 0;

int song[120];
int noteLength[120];

char serialInBuffer[bufferLength];
int serialCount = 0;

int i = 0;


// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

void uLCDprint(void){
  if(music == 0){
    uLCD.cls();
    uLCD.locate(1, 1);
    uLCD.printf("\nTwinkle Twinkle   \nLittle Star       \nLRLRUURL\n");
  }else if(music == 1){
    uLCD.cls();
    uLCD.locate(1, 1);
    uLCD.printf("\nTwo-Tigers        \n                  \nURLRUURU\n");
  }else if(music == 2){
    uLCD.cls();
    uLCD.locate(1, 1);
    uLCD.printf("\nLittle Bee        \n                  \nURLUUURR\n");
  }else{
    uLCD.cls();
    uLCD.locate(1, 1);
    uLCD.printf("\nError             \n                  \n");
  }
}

void DNN(void) {
  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  int gesture_index;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(), 1);

 // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return -1;
  }

  error_reporter->Report("Set up successful...\n");

  while (true) {

    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

    // Produce an output
    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
      if(mode == 1){
        if(gesture_index == 0 && selection == 0){
          if(music == 2){
            music = 0;  
          }else{
            music++;
          }
        }else if(gesture_index == 1 && selection == 0){
          if(music == 0){
            music = 2;
          }else{
            music--;
          }
        }else if(gesture_index == 2 && selection == 0){
          selection = 1;
          led1 = 0;
          led2 = 1;
          led3 = 1;
        }else if(gesture_index == 0 && selection == 1){
            music = 0;
            selection = 0;
            led1 = 1;
            led2 = 1;
            led3 = 1;
        }else if(gesture_index == 1 && selection == 1){
          music = 1;
          selection = 0;
          led1 = 1;
          led2 = 1;
          led3 = 1;
        }else if(gesture_index == 2 && selection == 1){
          music = 2;
          selection = 0;
          led1 = 1;
          led2 = 1;
          led3 = 1;
        }else{
          uLCD.locate(1, 1);
          uLCD.printf("\nerror             \n                  \n");
        }
      }

      else if(mode==0){

        if(i==8){
          uLCD.locate(0, 8);
          //uLCD.color(RED);
          uLCD.printf("Good!");
          i = 0;
        }

        if(gesture_index == 0 && i<8){
          //uLCD.cls();
          uLCD.locate(i, 7);
          uLCD.printf("R");
          i++;
        }

        else if(gesture_index == 1 && i<8){
          //uLCD.cls();
          uLCD.locate(i, 7);
          uLCD.printf("L");
          i++;
        }
        else if(gesture_index == 2 && i<8){
          //uLCD.cls();
          uLCD.locate(i, 7);
          uLCD.printf("U");
          i++;
        }
      
        
      

      }
    }
  }
}



void PlayMusic_and_TaiKu(void){
  mode = 1;
  led1 = 1;
  led2 = 1;
  led3 = 1;
  i = 0;
}

void Red_Mode(void){
  mode = 0;
  led1 = 1;
  led2 = 1;
  led3 = 1;
}



void playNote(int freq) {
  for(int i = 0; i < kAudioTxBufferSize; i++){
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  audio.spk.play(waveform, kAudioTxBufferSize);
}

void loadSignal(void)
{
  led2 = 0;
  int i = 0, k = 0;
  float freq = 0, len = 0;
  serialCount = 0;
  audio.spk.pause();
  uLCD.locate(1, 1);
  uLCD.printf("\nLoading Signal       \n                  \n");
  while(i < 120)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 5)
      {
        serialInBuffer[serialCount] = '\0';
        freq = (float) atof(serialInBuffer);
        song[i] = freq * 1000;
        serialCount = 0;
        i++;
      }
    }
  }
  while(k < 120)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 5)
      {
        serialInBuffer[serialCount] = '\0';
        len = (float) atof(serialInBuffer);
        noteLength[k] = len * 1000;
        serialCount = 0;
        k++;
      }
    }
  }
  led2 = 1;
}



int main(int argc, char* argv[]) {
  

  
  threadDNN.start(DNN);
  sw2.rise(PlayMusic_and_TaiKu);
  sw3.rise(Red_Mode);

  led3 = 1;
  led2 = 1;
  led1 = 1;

  loadSignal();

  while(1){
    uLCDprint();
    for(int i = 0; i < 40; i++){
      if(mode == 0){
        int length = noteLength[40 * music + i];
        while(length--){
        // the loop below will play the note for the duration of 1s
          for(int k = 0; k < kAudioSampleFrequency / kAudioTxBufferSize; ++k){
            playNote(song[40 * music + i]);
          }
        }
      }else{
        audio.spk.pause();
      }
    }
  }

}