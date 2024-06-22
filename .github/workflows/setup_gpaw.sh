export CIDERDIR=$PWD
cd ..
git clone https://gitlab.com/gpaw/gpaw.git
cd gpaw
cp $CIDERDIR/.github/workflows/gpaw_siteconfig.py .
python setup.py build
python setup.py install
cd ..
gpaw install-data --sg15 --register $PWD
gpaw install-data --register $PWD
cat ~/gpaw/rc.py
cd $CIDERDIR