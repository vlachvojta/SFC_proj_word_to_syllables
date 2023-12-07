# SFC_proj_word_to_syllables

SFC (soft computing) project at FIT (V|B)UT 2023/2024 winter semestr. Project containing material for creating AI model capable of inputing CZECH words split to syllables.

## Zadání
- původní funkční programy
- Python (pytorch)
- Přidat ukázkový běh (pokud programy berou nějaký parametry... nebo stačí default parametry asi...)
- Technická zpráva na 4-6 stran (včetně úvodní strany + všech dodatků...)
	- stučně popsat řešené problémy, stručné manuály pro překlad, spuštění...

- Install.sh
	- Udělat něco jako pip requirements...
	- Stáhnout data, instalovat knihovny atd (bez sudo)

- odevzdání:
	- 23-xvlach22.zip
	- kod i spustitelné soubory, technická zpráva v PDF
	- do 2 MB, nebo odkaz na Drive

- Obhajoba:
	- prezentace 7 minut (3-5 snímů WTF???) + diskuze 8 minut

## Co teda můžu udělat = vytvořit plán

- Finish dataset + Load (mini class)
- GRU Engine
	- n to n recurrent net (shifted by max(len(syllables)??)), s "konec slabiky" tokenem místo posledního písmenka daný slabiky
	- dvě varianty: moje from-scratch, pyTorch refference
- GRU trénování
	- evaluátor danýho modelu pro další trénování
		- použít pyTorch trénování i v mojí from-scratch variantě?
- Inference engine
	- Použít nějaký model, získat  "sl@b@ky" a převést na "sla-bi-ky"
- Simple GUI for showing
	- import training, GRU_engine...
	- zobrazí to:
		- grafiku GRU buňky a postupně hodnoty, jak tam cestujou
		- aktuální slovo, písmeno, stav modelu...
	- ovládání:
		- step-by-step, play with given speed, (return? - potřeba připravit engine a památování N posledních hodnot...)
	- viz "CEITEC" rešerše for py-gui (simple gui?)


## GRU torch starter-pack

docu:
https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
article with demo:
https://blog.floydhub.com/gru-with-pytorch/


## Co můžu udělat do konzultace

~~- Zapracovat na prezentaci?~~
- Zkusit matematicky aproximovat síť pro repeat mechanismus a začít trénovat od tama... (aspoň pochopím, jak to teda vevnitř funguje, looll...)
  - Zkusil jsem experiment, jestli zvládne vůbec síť se naučit opakování s 8 hidden_dims, nic moc, ehm...
    - Zkusit větší dims (+ novou strukturu fc, gru, fc ?)
- Zkusit větší síť, než jen jediná GRU vrstva => GDE trénovat? Zjistit PCSevcik. Zase Meta? blee...
  - Najít nějaký real zdroje, jak to dělal tamten týpek (crazy RNN, CNN, všechno možný, search Hyphenation)
- Pohekovat obecný algoritmus trénování pomocí Net definition (GRUNet...) + NetWrapper (spešl init, loss, optim...)
  - Zkusit CTCLoss? nebo prostě jinej typ chyby...
  - oddělat padding při trénování na stabilní hodnotu a dát tam dycky jen max(len(words))


Jak vylepšit síť:
- zkusit CTC Loss?
- dodělat embedding encoder + decoder
- zkusit změnit task na binární klasifikaci je/není konec slabiky (0-1 sigmoid aktivace?)
sentencepiece -> tokenizace
"NLLLoss je basically normal cross-entropie"

### SFC after konzultace: 

- dodělat, aby to fungovalo
- primárně torch
- ideálně zkusit vlastní GRU včetně chyby aaa

- v obhajobě se k tomu přiznat...

### obhajoba: 
 - mini úvod do tématu + teorie
- HLAVNĚ UKÁZAT funkčnost...
+ diskuze



## Co dál:
- otestovat pořádně pomocí testovací sady, zhodnotit
	- test.py
		- load model
		- iterate through words
		- get result, I guess...
- GUI appka?
	- ezy zadání slova, zobrazení výsledku + interaktivní zobrazení modelu?? (buňky, a písmenka jako v docu)
- na to GRU by hand se asi vykáknu, max udělám něco basic, jako proof of no working concept...


