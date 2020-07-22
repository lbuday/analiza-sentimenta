Reviewovi se moraju downloadat sa stranice na linku: "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
Automatski se sinu i extractaju, newline se zamjeni sa razakom, makne se interpunkcija,
maknu se svi dupli razmaci i pretvorimo u array.

Skinemo vektorsko mapiranje rijeci sa linka: "http://nlp.stanford.edu/data/glove.6B.zip"
Pomoću mapiranja pretovrimo rijeci u intove. Odstranimo predugačke reviewove,
a prekratke paddamo sa nulama

Podatke sacuvamo(pickelamo) za sljedeci puta