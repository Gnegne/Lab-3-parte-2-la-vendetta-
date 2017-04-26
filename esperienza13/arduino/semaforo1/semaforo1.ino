// prova n.1 di macchina a stati finiti semaforo
//avtore: Thomas Giannoni
//non ancora verificato
//vltima modifica: 25/04/2017

#define INP 8   //il nostro abilitatore
#define LED1 9  //LED verde  
#define LED2 10  //LED giallo
#define LED3 11  //LED rosso

// le variabili che ho usato
int val = 0;
int t = 0;

void setup() {
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT);
  pinMode(INP, INPUT_PULLUP);
}

void loop() {
  val = digitalRead(INP);
  if(val == HIGH) {
    while(t<1000){
      digitalWrite(LED2, HIGH);
      delay(100);
      digitalWrite(LED2, LOW);
      delay(100);
      t = t + 200;
    }
    t = 0;
   }
   else{
    digitalWrite(LED1, HIGH);
    delay(1000);
    val = digitalRead(INP);
    if(val == HIGH) {
        digitalWrite(LED1, LOW);
        return;
    }
    digitalWrite(LED2, HIGH);
    delay(1000);
    val = digitalRead(INP);
    digitalWrite(LED1, LOW);
    digitalWrite(LED1, LOW);
     if(val == HIGH) {
      return;
    }
    digitalWrite(LED3, HIGH);
    delay(1000);
    digitalWrite(LED3, LOW);
   }
}

