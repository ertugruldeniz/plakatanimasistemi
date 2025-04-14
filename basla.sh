#!/bin/bash

echo "Plaka Tanima Sistemi baslatiliyor..."
echo ""

# Python kurulu mu kontrol et
if ! command -v python3 &> /dev/null; then
    echo "Python3 kurulu degil! Lutfen Python'u yukleyin."
    echo "Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "Fedora: sudo dnf install python3 python3-pip"
    echo ""
    exit 1
fi

# Çalıştırma izni ver
chmod +x basla.py

# Uygulamayı başlat
python3 basla.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Bir hata olustu."
fi 