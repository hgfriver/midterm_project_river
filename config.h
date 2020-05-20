#ifndef CONFIG_H_
#define CONFIG_H_

// The number of labels (without negative)
#define label_num 3

struct Config {

  // This must be the same as seq_length in the src/model_train/config.py
  const int seq_length = 64;

  // The number of expected consecutive inferences for each gesture type.
  const int consecutiveInferenceThresholds[label_num] = {20, 20, 15};

  const char* output_message[label_num] = {
        "RIGHT:\n\r"
        "                  \n\r"
        "                  \n\r"
        "                  \n\r"
        "    ->->->->->    \n\r"
        "                  \n\r"
        "                  \n\r"
        "                  \n\r",
        "LEFT:\n\r"
        "                 \n\r"
        "                 \n\r"
        "                 \n\r"
        "    <-<-<-<-<-   \n\r"
        "                 \n\r"
        "                 \n\r"
        "                 \n\r"
        "                 \n\r"
        "v:\n\r"        
        "        ^         \n\r"
        "        |         \n\r"
        "        ^         \n\r"
        "        |         \n\r"
        "        ^         \n\r"
        "        |         \n\r"
        "        ^        \n\r"
        "        |         \n\r"};
};

Config config;
#endif // CONFIG_H_