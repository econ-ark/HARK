if [[ $TRAVIS_PYTHON_VERSION == 3.6 ]]; then
    pytest --nbval-lax DemARK/notebooks
    pytest --nbval-lax REMARK/REMARKs
fi