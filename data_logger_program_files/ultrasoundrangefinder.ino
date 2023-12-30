// logs ultrasound distance and transmits it over the serial. Uses a switch to pause or resume the data log to generate new files.

// defines pins numbers
const int trigPin = 9;
const int echoPin = 10;
const int switchPin = 11;
// defines variables
long duration;
int distance;
int switchState;
bool datalogging = false;
void setup() {
  pinMode(trigPin, OUTPUT); // Sets the pin modes
  pinMode(echoPin, INPUT);
  pinMode(switchPin,INPUT_PULLUP); 
  Serial.begin(9600); 
}


void loop() {
  switchState = digitalRead(switchPin);
  if(switchState == HIGH){
    if(datalogging){
      Serial.println("ENDITRIGHTNOW");
    }
    else{
      Serial.println("GOGOGOGOGO");
    }
    delay(500);
  }
  else{
  // Clears the trigPin
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2); 
  // Sets the trigPin on HIGH state for 10 micro seconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);
  // Calculating the distance
  distance = duration * 0.034 / 2;
  // Prints the distance on the Serial Monitor
  Serial.println(distance);
  }
}
