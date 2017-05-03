// prova n.2 di macchina a stati finiti semaforo con buzzer che sona su giallo 
//autore: Thomas Giannoni Roberto Ribatti
//ultima modifica: 02/05/2017

#define INP 12   //il nostro abilitatore
#define LED1 9  //LED verde  
#define LED2 10  //LED giallo
#define LED3 11  //LED rosso

int timev = 10;
int timeg = 3;
int timer = 10;
int timegen = 1;
int clockk = 500;
int stato = 0;
int tprec = 0;
int tnow = 0;
int tstato=0;
int val = 0;
//stato 0 = verde
//stato 1 = giallo verde
//stato 2 = rosso
//stato 3 = giallo
//stato 4 = spento

void setup() {
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT);
  pinMode(INP, INPUT_PULLUP);
}

void loop() {
  // le variabili che ho usato
  tnow = millis();
  if(tnow-tprec >clockk){
    val = digitalRead(INP);
    
    if(val == HIGH & stato == 3 & tnow-tstato > timegen*clockk) {
      stato = 4;
      output(stato);
      tstato = tnow;
    }
    if(val == HIGH & stato == 4 & tnow-tstato > timegen*clockk) {
      stato = 3;
      output(stato);
      tstato=tnow;
    }
    if(val == HIGH & stato != 4 & stato !=3) {
      stato = 3;
      output(stato);
      tstato=tnow;
    }
    if(val == LOW & stato == 0 & tnow-tstato > timev*clockk) {
      stato = 1;
      output(stato);
      tstato=tnow;
    }
    if(val == LOW & stato == 1 & tnow-tstato > timeg*clockk) {
      stato = 2;
      output(stato);
      tstato=tnow;
    }
    if(val == LOW & stato == 2 & tnow-tstato > timer*clockk) {
      stato = 0;
      output(stato);
      tstato=tnow;
    }
    if(val == LOW & stato != 0 & stato !=1 & stato!=2) {
      stato = 0;
      output(stato);
      tstato=tnow;
    }
  tprec=tnow;
  }
}

void output(int stato){
if (stato == 0){
   digitalWrite(LED1, HIGH);
   digitalWrite(LED2, LOW);
   digitalWrite(LED3, LOW);
  }
if (stato == 1){
   digitalWrite(LED1, HIGH);
   digitalWrite(LED2, HIGH);
   digitalWrite(LED3, LOW);
  }    
if (stato == 2){
   digitalWrite(LED1, LOW);
   digitalWrite(LED2, LOW);
   digitalWrite(LED3, HIGH);
  }
if (stato == 3){
   digitalWrite(LED1, LOW);
   digitalWrite(LED2, HIGH);
   digitalWrite(LED3, LOW);
  } 
  if (stato == 4){
   digitalWrite(LED1, LOW);
   digitalWrite(LED2, LOW);
   digitalWrite(LED3, LOW);
  }
}   
    
    
    
    
    
    

