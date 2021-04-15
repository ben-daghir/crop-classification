import os

# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=25, fill='█', printEnd="\r"):
    """
    @source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def mail(receiver_emails, message):
    import smtplib, ssl

    receiver_emails = ['bdaghir@gatech.edu', 'lderado3@gatech.edu', 'emccaskey@gatech.edu', 'khangvu@gatech.edu']
    port = 587  # For starttls
    smtp_server = "smtp.gmail.com"
    sender_email = "eduroam.pi.ip@gmail.com"
    password = os.environ['G_PW']

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port, None, 30) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        for receiver_email in receiver_emails:
            print("Emailing: " + str(receiver_email))
            server.sendmail(sender_email, receiver_email, message)
