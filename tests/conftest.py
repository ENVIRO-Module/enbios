import os

import pytest
import sys
import datetime
import time
import ctypes.util
from typing import Union

# scopes (broader to narrower): session, module, class, function (default)


@pytest.fixture(scope="session")
def chrome_driver():
    """
    Prepare Chrome driver, visible or headless
    :return:
    """
    driver = None  # webdriver.Chrome(options=options)

    yield driver  # Execute tests

    driver.quit()


@pytest.fixture(scope="session")
def mssql_database():
    connection = None  # pyodbc.connect(connect_string)

    yield connection

    connection.close()


@pytest.fixture(scope="session")
def h2_database():
    database = None
    yield database
    # TODO Close connection


@pytest.fixture(scope="function")  # Each test function
def time_set():
    """
    Fixture to KEEP system date after a function assumed to set the LOCAL system date
    This would be used by test setting a specific datetime

    :return:
    """

    t1 = datetime.datetime.now()

    yield t1  # Test execution

    # Reset time (considering the time that the test took to execute)
    dif = datetime.datetime.now() - t1

    # Recover system time
    # set_system_date(t1 + dif)


@pytest.fixture(scope="session")
def base_data():
    # TODO Load base data (using dbUnit or other)
    return


