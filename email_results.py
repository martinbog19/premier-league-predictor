import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pretty_html_table import build_table


def send(CONTENT, subject = None) :

    # Specify the email contents
    mail = MIMEMultipart()
    html = """\
    <html><head></head><body>{0}</body></html>
    """.format(build_table(CONTENT, 'grey_light', text_align = 'right', font_family = 'arial', width_dict = ['100','200','200','100','100','100','100'], font_size = 10))
    mail.attach(MIMEText(html, 'html'))

    # Set my email address and the password key
    my_mail  = 'martinbog19@gmail.com'
    with open('gmail_key.txt') as f:
        password = f.read()
    # Set the subject of the email
    mail['Subject'] = subject

    # Connect to the Gmail SMTP server & log in
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(my_mail, password)

    # Send the email (to myself)
    server.sendmail(my_mail, my_mail, mail.as_string())
    q = server.quit()