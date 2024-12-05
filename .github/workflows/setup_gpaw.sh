export CIDERDIR=$PWD
cd ..
git clone https://gitlab.com/gpaw/gpaw.git
cd gpaw
cp $CIDERDIR/.github/workflows/gpaw_siteconfig.py siteconfig.py
pip install .
cd ..
gpaw install-data --sg15 --register $PWD
gpaw install-data --register $PWD
cd $CIDERDIR
