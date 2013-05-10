cudaQt-reference-project
========================

Załączony projekt ma służyć sprawdzeniu, czy CUDA/Qt działają "prosto z pudełka" z projektami utworzonymi na innej maszynie.

Projekt ten korzysta z templatowego projektu CUDA wzbogaconego o Qt.
Aby sprawdzić funkcjonalność obu bibliotek program otwiera plik graficzny zaszyty w kodzie i zapisuje go pod podaną ścieżką (również hardcode'owana w kodzie), a następnie uruchamia jakiś kernel CUDA.

Projekt powstał poprzez:

1. stworzenie nowego projektu CUDA z template'a korzystając z Wizarda;
2. przekształcono projekt na projekt Qt korzystając z Qt-Addin do VC oraz tego: http://stackoverflow.com/questions/2088315/how-to-convert-a-regular-win32-vc-vcproj-project-to-a-qt-project rozwiązania (pierwsza odpowiedź)
3. zmieniono ustawienia projektu Qt aby korzystał z GUI (ppm na projekcie w Solution Explorer w VC -> Qt options)
4. dodano: "$(QTDIR)\lib" do "Additional Library Directories" w opcjach Linker'a w opcjach projektu (ppm na projekcie -> properties)
5. do "Additional Include Directiories" w opcjach CUDA C++ w opcjach projektu dodano:<br>
  ".<br>
  $(QTDIR)\include<br>
  .\GeneratedFiles\$(ConfigurationName)<br>
  $(QTDIR)\include\QtCore<br>
  $(QTDIR)\include\QtGui"<br>
6. do kodu dodano:<br>
  "QImage img;<br>
  img.load("sciezka-do-obrazka");<br>
  img.save("sciezka-gdzie-zapisac-obrazek");"<br>
	
U mnie działa. Dajcie znać jak u was.

p.s. ja instalowałem narzędzia w kolejności:

1. Visual Studio 2010 Pro
2. Qt VC
3. Qt-Addin VC
4. CUDA Toolkit

p.p.s. wklejano bez cudzysłowów a ścieżkę podano konkretną ;)
