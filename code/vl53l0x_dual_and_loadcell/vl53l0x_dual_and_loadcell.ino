//include libraries needed 
#include "Adafruit_VL53L0X.h"
#include "HX711.h"

// address we will assign if dual sensor is present
#define LOX1_ADDRESS 0x30
#define LOX2_ADDRESS 0x31

// set the pins to shutdown
#define SHT_LOX1 3 // tof 1 
#define SHT_LOX2 2 // tof 2 

// objects for the vl53l0x
Adafruit_VL53L0X lox1 = Adafruit_VL53L0X();
Adafruit_VL53L0X lox2 = Adafruit_VL53L0X();
HX711 scale;

// this holds the measurement
VL53L0X_RangingMeasurementData_t measure1;
VL53L0X_RangingMeasurementData_t measure2;

// variables needed 
uint16_t tof1;  // holds the tof reading for tof1 
uint16_t tof2; // holds the tof reading for tof2 
const int LOADCELL_DOUT_PIN = 10; //pin for the load cell data wire 
const int LOADCELL_SCK_PIN = 11; // pin for the load cell clock wire 
float lastReading = 0.0;// holds last load cell reading values 

// USED GIVEN FUNCTION IN Adafruit_VL53L0X Libary 
/*
    Reset all sensors by setting all of their XSHUT pins low for delay(10), then set all XSHUT high to bring out of reset
    Keep sensor #1 awake by keeping XSHUT pin high
    Put all other sensors into shutdown by pulling XSHUT pins low
    Initialize sensor #1 with lox.begin(new_i2c_address) Pick any number but 0x29 and it must be under 0x7F. Going with 0x30 to 0x3F is probably OK.
    Keep sensor #1 awake, and now bring sensor #2 out of reset by setting its XSHUT pin high.
    Initialize sensor #2 with lox.begin(new_i2c_address) Pick any number but 0x29 and whatever you set the first sensor to
 */
void setID() {
  // all reset
  digitalWrite(SHT_LOX1, LOW);    
  digitalWrite(SHT_LOX2, LOW);
  delay(10);
  // all unreset
  digitalWrite(SHT_LOX1, HIGH);
  digitalWrite(SHT_LOX2, HIGH);
  //delay(10);

  // activating LOX1 and resetting LOX2
  digitalWrite(SHT_LOX1, HIGH);
  digitalWrite(SHT_LOX2, LOW);

  // initing LOX1
  if(!lox1.begin(LOX1_ADDRESS)) {
    Serial.println(F("Failed to boot first VL53L0X"));
    while(1);
  }
  delay(10);

  // activating LOX2
  digitalWrite(SHT_LOX2, HIGH);
  delay(10);

  //initing LOX2
  if(!lox2.begin(LOX2_ADDRESS)) {
    Serial.println(F("Failed to boot second VL53L0X"));
    while(1);
  }
}

uint16_t read_dual_sensors() {
  
  lox1.rangingTest(&measure1, false); // pass in 'true' to get debug data printout!
  lox2.rangingTest(&measure2, false); // pass in 'true' to get debug data printout!

  // print sensor one reading
  Serial.print(F("1; "));
  if(measure1.RangeStatus != 4) {     // if not out of range
    Serial.print(measure1.RangeMilliMeter);
    tof1 = measure1.RangeMilliMeter; 
  } else {
    Serial.print(F("0"));
     tof1 = 0; 
  }
  
  Serial.print(F("; "));

  // print sensor two reading
  Serial.print(F("2; "));
  if(measure2.RangeStatus != 4) {
    Serial.print(measure2.RangeMilliMeter);
    tof2 = measure2.RangeMilliMeter ;
  } else {
    Serial.print(F("0"));
    tof2 = 0;
  }
  
  Serial.println();
  return tof1, tof2;
}

void setup() {
  // start serial 
  Serial.begin(115200);

  // wait until serial port opens for native USB devices
  while (! Serial) { delay(1); }

  // set pin modes 
  pinMode(SHT_LOX1, OUTPUT);
  pinMode(SHT_LOX2, OUTPUT);

  // set pins to low 
  digitalWrite(SHT_LOX1, LOW);
  digitalWrite(SHT_LOX2, LOW);
  
  setID();
  
  startLoadCell();

  //get first reading 
  float loadCellReading = scale.get_units();  
  lastReading = loadCellReading;
}

void loop() {
  //https://arduino.stackexchange.com/questions/22272/how-do-i-run-a-loop-for-a-specific-amount-of-time
  
  //get tof sensor readings 
  tof1, tof2 = read_dual_sensors();

  //get load cell reading
  float loadCellReading = scale.get_units();    
  //Serial.println(loadCellReading);
  if (loadCellReading - lastReading > 50) // if the load cell reading is 5 N higher than the last load cell reading 
  {
    // print TOUCH 3 imes to make sure that it is seen 
    Serial.println("TOUCH");
    Serial.println("TOUCH");
    Serial.println("TOUCH");
    //exit(0);
  }

  lastReading = loadCellReading; 
  
  delay(100);
  
}
void startLoadCell() {
  // function from Chelse 
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
            
  scale.set_scale(433);        // this value is obtained by calibrating the scale with known weights; see the README for details
  scale.tare();               // reset the scale to 0
}
