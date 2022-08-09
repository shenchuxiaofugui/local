import time
from splinter import Browser


def login_mail(url):
    browser = Browser()
    # login 163 email websize
    browser.visit(url)
    # wait web element loadloginBting
    # fill in account and password
    browser.find_by_id('username').fill('51214700053')
    browser.find_by_id('password').fill('321juexing!')
    # click the button of login
    browser.find_by_id('submit').click()
    time.sleep(5)
    # close the window of brower
    browser.quit()


if __name__ == '__main__':
    mail_addr = 'https://login.ecnu.edu.cn/'
    login_mail(mail_addr)