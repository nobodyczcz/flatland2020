
# You need to install p2j to convert the python scripts into notebooks
# pip install p2j

for sPyFile in ./examples/*.py
do
    # make the path for the destination file
    # swap examples for notebooks
    sNBFile="${sPyFile/examples/notebooks}"
    # swap .py for .ipynb
    sNBFile="${sNBFile/.py/.ipynb}"

    # convert the python file $sPyFile into a notebook $sNBFile
    # -o: overwrite any existing notebook file
    echo p2j ${sPyFile} -t ${sNBFile} -o

done


# p2j ./examples/introduction_flatland_2_1.py -t ./notebooks/introduction_flatland_2_1.ipynb -o