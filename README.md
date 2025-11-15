<strong>Detectia glaucomului</strong>
<hr>
Acest repository contine un cod in MATLAB care foloseste algoritmul de Cup to Disc Ratio (CDR) pentru a determina glaucomul pe baza unor imagini din sursa ORIGA realizata de specialisti. 
<br>
Codul foloseste o baza de date OrigaList ce contine date despre fiecare imagine si T/F pentru diagnostic, lucru posibil cu ajutorul unor masti marcate manual pe baza pozelor cu fundus al ochiului realizate de specialisti.
<br>
Codul calculeaza si gradul de eficienta cu ajutorul Receiver Operating Characteristic (ROC) curb si afiseaza trei grafice pentru fiecare CDR calculat.
<br>
Ca si output, codul genereaza un fisier .csv ce contine un tabel cu datele despre fiecare imagine si diagnosticul corespunzator, verificat cu datele din OrigaList.mat.
<br>
<br>
<strong>Cerinte laborator 2: </strong>
<ul>
  <li>Link analiza literaturii de specialitate: https://docs.google.com/document/d/149VfS16jrfrd9MImClk6J7S7mb6FnjpagzVOxpnH4qs/edit?usp=sharing</li>
  <li>Link proiectarea soluției/aplicației: https://docs.google.com/drawings/d/1xrWu2dzmHSkYJ-WAHXXUPsx4eFBkrtnrVlTxz-C7Pek/edit?usp=sharing</li>
</ul>
In schema bloc se pot vedea cele 2 etape, training si testing, cu pasi principali precum extragerea datelor din imaginile cu fundus, pre-procesare, crearea bazei de date si in final returnarea rezultatului dorit.
<br>
<br>
<strong>Sursa datelor si imaginilor provin din arhiva Kaggle: https://www.kaggle.com/datasets/sshikamaru/glaucoma-detection </strong>
<hr>


[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/BzgEFjMi)
