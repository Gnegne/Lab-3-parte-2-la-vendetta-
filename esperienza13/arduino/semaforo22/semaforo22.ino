// Traffic light FSM on ARduino

#define INP      12    // pin enable
#define LED1     9     // pin green led  
#define LED2     10    // pin yellow led
#define LED3     11    // pin red led

#define timeg    10   // green light duration in clock period
#define timey    3    // yellow light duration in clock period
#define timer    10   // red light duration in clock period
#define timeen   1    // yellow/off duration in clock period
#define clockk   500  // clock period in ms

/*
  state 0 => green light
  state 1 => green and yellow light at the same time
  state 2 => red light
  state 3 => yellow light
  state 4 => switched-off
*/

int LED[3] = {LED1, LED2, LED3};
int time_state[5] = {timeg, timey, timer, timeen, timeen};
int state = 0;
int out[5][3] = {{1,0,0},{1,1,0},{0,0,1},{0,1,0},{0,0,0}};

int tnow = 0;   // current time
int tprec = 0;  // time since the last click of the clock
int tstate = 0; // time since the last state change
bool enable = HIGH;


void setup() {
  // Initialization pin mode
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT);
  pinMode(INP, INPUT_PULLUP);
}

void loop() {
  tnow = millis();
  if(ClockEnd(tnow, tprec)){ // verify if a clock period has passed
    enable = digitalRead(INP);  // reading the enable state
    if (EndState(state, tnow, tstate)){ // verify if the current state time has ended
      state = NextState(state, enable); // switch to the next state
      Output(state);  // generate output
      tstate = tnow;  //setting the starting time of the current state
    }
  tprec=tnow; //setting the click time
  }
}

bool ClockEnd(int tnow, int tprec){ // TRUE if a clock period has passed
  return tnow-tprec > clockk;
}

bool EndState(int state, int tnow, int tstate){ // TRUE if the current state has ended
return tnow-tstate > time_state[state]*clockk;
}

int NextState(int state, bool enable){ //calculate next state from current state and enable level
  if(enable) return (state+1)%3;  // when the enable is HIGH loop between 0, 1, 2
  else  return 3+ state%2;  // when enable is LOW loop between 3 and 4
}

void Output(int stato){ // generate output from the state
  for (int j=0; j<3; j++){
    digitalWrite(LED[j], out[state][j]);
  }
}
