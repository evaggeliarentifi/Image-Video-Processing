import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import skimage.util
from skimage.morphology import disk
from skimage import filters

# Για το ερώτημα 1
# συνάρτηση που διαβάζει όλο το βίντεο
# μείωνει την ανάλυση στο μισό και επιλέγει τον κατάλληλο χρωματικό χώρο
def video(cap):
    while cap2.isOpened():
        ret, frame = cap.read()
        height, width = frame.shape[0:2]
        frame = cv2.resize(frame, (width // 2, height // 2), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        # μετατροπή σε grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("A simple video player", gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Για ερώτημα 4
# η συνάρτηση δέχεται σαν όρισμα το προς επεξεργασία βίντεο,
# τα σημεία ενδιαφέροντος(που έχουν ανιχνευθεί με harris corner ή shi tomasi detector)
#  και τα βασικά ορίσματα της συνάρτησης  cv2.calcOpticalFlowPyrLK  που υπολογίζει το optical flow
#  συμφωνα με τον αλγόριθμο Lucas Kanade

def Lucas_Kanade(cap, corners, win_size, max_level, crit):
    # παράμετροι για Lucas Kanade optical flow
    lk_params = dict(winSize=win_size, maxLevel=max_level, criteria=crit)

    color = (0, 0, 255)

    # διαβάζουμε το πρώτο frame και ρίχνουμε την ανάλυση στο μισό,
    # όπως ζητείται από την εκφώνηση της άσκησης
    ret, first_frame = cap.read()
    height, width = first_frame.shape[0:2]
    first_frame = cv2.resize(first_frame, (width // 2, height // 2), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    # μετατρέπουμε το πρώτο frame σε grayscale αφού
    # χρειαζόμαστε μονο το κανάλι φωτεινότητας(luminance channel)
    # για την ανίχνευση των γωνιών και επίσης έχει μικρότερο υπολολιστικό κόστος
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # θέτουμε prev ίσο με τα σημεία ενδιαφέροντος που εντοπίστηκαν
    # στο πρώτο frame,ώστε να παρακαολουθήσουμε την ροή αυτών των σημείων
    prev = corners

    # Δημιουργούμε μια εικόνα γεμάτη μηδενικές εντάσεις με τις ίδιες διαστάσεις με το πρώτο frame -
    # για μελλοντικούς  σκοπούς σχεδίασης
    mask = np.zeros_like(first_frame)

    while (cap.isOpened()):

        # μειώνουμε την ανάλυση κάθε frame του βίντεο στο μισό
        # και μετατρέπουμε σε grayscale
        ret, frame = cap.read()
        frame = cv2.resize(frame, (width // 2, height // 2), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # υπολογίζουμε την αραιά οπτική ροή με
        # Lucas Kanade
        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)

        # επιλέγουμε τα σημέια ενδιαφέροντος για την προηγούμενη θέση
        good_old = prev[status == 1]

        # επιλέγουμε τα σημεία ενδιαφέροντος για την επόμενη θέση
        good_new = next[status == 1]

        # ζωγραφίζουμε την οπτική ροή
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # a,b οι συντεταγμένες του νέου σημείου
            a, b = np.int0(new.ravel())

            # c,d οι συντεταγμενες του παλιού σημείου
            c, d = np.int0(old.ravel())

            # ζωγραφίζουμε την γραμμή μεταξύ της παλιάς και της καινούργιας θέσης
            # με κόκκινο χρώμα και πάχος 1
            mask = cv2.line(mask, (a, b), (c, d), color, 1)

            # ζωγραφίζουμε ένα "γεμάτο" κύκλο στη νέα θέση με ακτίνα 2
            frame = cv2.circle(frame, (a, b), 2, color, -1)

        # επικαλύπτει τα ίχνη οπτικής ροής στο αρχικό πλαίσιο
        output = cv2.add(frame, mask)

        # ενημερώνει το previous frame
        prev_gray = gray.copy()

        # ενημερώνει τα previous good features points
        prev = good_new.reshape(-1, 1, 2)

        # ανοίγουμε νέο παράθυρο και εμφανίζουμε το αποτέλεσμα
        cv2.imshow("sparse optical flow", output)

        # το πρόγραμμα βγαίνει απο το while loop όταν ο χρήστης πατήσει το 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # απελευθερώνουμε πόρους και κλείνουμε τα παράθυρα
    cap.release()
    cv2.destroyAllWindows()


# Για ερώτημα 5

# η παρακάτω συνάρτηση παίρνει σαν όρισμα  2 πίνακες
# και κρατάει μόνο  στοιχεία  που απέχουν περισσοτερο απο 1
# όπως ζητείται απο την εκφώνηση της άσκησης
# το σκεπτικό είναι ότι κάθε γωνία που δεν μετακινείται είναι
# και στους 2 πίνακες
def good_corners(newCorners, oldCorners):

    # κρατάμε το μήκος των δύο πινάκων
    newCornersLength = len(newCorners)
    oldCornersLength = len(oldCorners)

    # δημιουργούμε έναν πίνακα για να κρατήσουμε
    # το αποτέλεσμα, δλδ τις τελικές γωνίες
    final_corners = []

    counterN = 0
    counterO = 0

    while counterN < newCornersLength and counterO < oldCornersLength:
        if np.absolute(newCorners[counterN][0][0] - oldCorners[counterO][0][0]) < 1 and np.absolute(
                newCorners[counterN][0][1] - oldCorners[counterO][0][1]) < 1:
            counterN += 1
            counterO += 1
        else:
            final_corners.append(newCorners[counterN])
            counterN += 1
    for i in range(counterN, newCornersLength):
        final_corners.append(newCorners[i])
    return np.array(final_corners)


# η παρακάτω συνάρτηση που αποτλεί μια εξέλιξη
# της συνάρτησης του ερωτήματος 5,παίρνει σαν ορίσματα το προς επεξεργασία βίντεο,
# τα βασικά ορίσματα της συνάρτησης  cv2.calcOpticalFlowPyrLK  που υπολογίζει το optical flow
# συμφωνα με τον αλγόριθμο Lucas Kanade , καθως και τα βασικά ορίσματα της συνάρτησης
# cv2.goodFeaturesToTrack για την ανίχνευση των ζητούμενων γωνιών-σημείων ενδιαφέροντος


def lucas_kanade_5(cap, lk_param, corner_param):
    framesForCheck = 20

    # αρχικοποιούμε το χρώμα με το οποίο θα ζωγραφιστούν
    # οι γραμμές που θα δείχνουν την οπτική ροή
    color = (0, 0, 255)

    # διαβάζουμε το 20ο frame
    for i in range(framesForCheck):
        reti, test_frame = cap.read()

    # μειώνουμε την ανάλυση στο μισό,όπως απαιτείται
    # απο το 1ο βήμα της άσκησης
    height, width = test_frame.shape[0:2]
    test_frame = cv2.resize(test_frame, (width // 2, height // 2), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    # μετατρέπουμε σε grayscale
    prev_gray2 = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)

    # διαβάζουμε το 1ο frame και κάνουμε πάλι το
    # απαραίτητο resize
    ret, first_frame = cap.read()
    height, width = first_frame.shape[0:2]
    first_frame = cv2.resize(first_frame, (width // 2, height // 2), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    # αρχικοποιούμε τον μετρητή που μας δείχνει
    # σε ποιο frame είμαστε κάθε φορά
    counter = 0

    # μετατρέπουμε σε grayscale το 1ο frame
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # βρίσκουμε τα σημεία ενδιαφέροντος-γωνίες στο 1ο και στο 25ο frame
    prev = cv2.goodFeaturesToTrack(prev_gray, **corner_param)
    next2 = cv2.goodFeaturesToTrack(prev_gray2, **corner_param)

    # στην μεταβλητή starting_corners σώζουμε τις γωνίες του 1ου frame
    starting_corners = prev

    # διαγράφουμε τις γωνίες που δεν έχουν αλλάξει θέση απο το 1ο
    # στο 25ο frame , με την χρήση της συνάρτησης good_corners
    # που παρατίθεται παραπάνω
    prev = good_corners(prev, next2)

    # φτιάχνουμε μια μάσκα με μηδενικα, με μέγεθος όσο και το 1ο frame
    mask = np.zeros_like(first_frame)

    while cap.isOpened():

        # διάβασμα και resize του frame
        ret, another_frame = cap.read()
        height, width = another_frame.shape[0:2]
        another_frame = cv2.resize(another_frame, (width // 2, height // 2), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        if ret == True:

            # μετατροπή σε grayscale
            gray = cv2.cvtColor(another_frame, cv2.COLOR_BGR2GRAY)

            # υπολογίζουμε την οπτική ροή με τον αλγόριθμο Lukas Kanade
            next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_param)

            # επιλέγουμε τα σημεία ενδιαφέροντος για την προηγούμενη θέση
            good_old = prev[status == 1]

            # επιλέγουμε τα σημεία ενδιαφέροντος για την επόμενη θέση
            good_new = next[status == 1]

            # ζωγραφίζουμε την οπτική ροή
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                # a,b οι συντεταγμένες του νέου σημείου
                a, b = np.int0(new.ravel())

                # c,d οι συντεταγμένες του παλιού σημείου
                c, d = np.int0(old.ravel())

                # ζωγραφίζουμε την γραμμή μεταξύ της παλιάς και της καινούργιας θέσης
                # με κόκκινο χρώμα και πάχος 1
                mask = cv2.line(mask, (a, b), (c, d), color, 1)

                # ζωγραφίζουμε ένα "γεμάτο" κύκλο στη νέα θέση με ακτίνα 2
                third_frame = cv2.circle(another_frame, (a, b), 2, color, -1)

            # επικαλύπτει τα ίχνη οπτικής ροής στο αρχικό πλαίσιο
            output = cv2.add(another_frame, mask)

            # ενημερώνει το previous frame
            prev_gray = gray.copy()

            # ενημερώνει τα previous good features points
            prev = good_new.reshape(-1, 1, 2)

            # ανοίγουμε νέο παράθυρο  και εμφανίζουμε το αποτέλεσμα
            cv2.imshow("Lukas Kanade_5", output)

            # για κάθε 20 frames προσθέτουμε μόνο τις γωνίες που μετακινούνται
            if counter == framesForCheck:
                # οταν διαβάσουμε 20 frames , ο μετρητής ξανααρχικοποιείται στο 0
                counter = 0
                prev_gray = gray
                previus_corners = prev
                prev = cv2.goodFeaturesToTrack(prev_gray, **corner_param)

                # ελέγχουμε πάλι ποιες γωνίες κραταμε
                prev = good_corners(prev, starting_corners)
                prev = good_corners(prev, previus_corners)

            # αύξηση μετρητή κατα 1 για να ξέρουμε σε ποιο frame είμαστε
            counter += 1
        else:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    # απελευθερώνουμε πόρους και κλείνουμε τα παράθυρα
    cap.release()
    cv2.destroyAllWindows()

# Για ερώτημα 6

# δημιουργία της συνάρτησης snpAmount η οποία θα χρειαστεί
# για τον καθορισμό της τιμής του amount στην προσθήκη
# του θορύβου salt&pepper στην εικόνα

def snpAmount(x):
    amount = x / 90 + 0.3
    return amount

def lucas_kanade_noise(cap, lk_param, corner_param):
    framesForCheck = 20

    # αρχικοποιούμε το χρώμα με το οποίο θα ζωγραφιστούν
    # οι γραμμές που θα δείχνουν την οπτική ροή
    color = (0, 0, 255)

    # διαβάζουμε το 20ο frame
    for i in range(framesForCheck):
        reti, test_frame = cap.read()

    # μειώνουμε την ανάλυση στο μισό,όπως απαιτείται
    # απο το 1ο βήμα της άσκησης
    height, width = test_frame.shape[0:2]
    test_frame = cv2.resize(test_frame, (width // 2, height // 2), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    # προσθήκη θορύβου στο 20ο frame
    test_frame = skimage.util.random_noise(test_frame, mode='s&p', seed=4, amount=snpAmount(6))
    test_frame = np.array(255 * test_frame, dtype='uint8')

    # μετατρέπουμε σε grayscale
    prev_gray2 = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)

    # διαβάζουμε το 1ο frame και κάνουμε πάλι το
    # απαραίτητο resize
    ret, first_frame = cap.read()
    height, width = first_frame.shape[0:2]
    first_frame = cv2.resize(first_frame, (width // 2, height // 2), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    # προσθήκη θορύβου στο 1ο frame
    first_frame = skimage.util.random_noise(first_frame, mode='s&p', seed=4, amount=snpAmount(6))
    first_frame = np.array(255 * first_frame, dtype='uint8')


    # αρχικοποιούμε τον μετρητή που μας δείχνει
    # σε ποιο frame είμαστε κάθε φορά
    counter = 0

    # μετατρέπουμε σε grayscale το 1ο frame
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # βρίσκουμε τα σημεία ενδιαφέροντος-γωνίες στο 1ο και στο 20ο frame
    # ενώ έχει προστεθέι θόρυβος
    prev = cv2.goodFeaturesToTrack(prev_gray, **corner_param)
    next2 = cv2.goodFeaturesToTrack(prev_gray2, **corner_param)

    # στην μεταβλητή starting_corners σώζουμε τις γωνίες του 1ου frame
    starting_corners = prev

    # διαγράφουμε τις γωνίες που δεν έχουν αλλάξει θέση απο το 1ο
    # στο 20ο frame , με την χρήση της συνάρτησης good_corners
    # που παρατίθεται παραπάνω
    prev = good_corners(prev, next2)

    # φτιάχνουμε μια μάσκα με μηδενικα, με μέγεθος όσο και το 1ο frame
    mask = np.zeros_like(first_frame)

    while cap.isOpened():

        # διάβασμα και resize του frame
        ret, another_frame = cap.read()
        height, width = another_frame.shape[0:2]
        another_frame = cv2.resize(another_frame, (width // 2, height // 2), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        if ret == True:

            # όπως ζητέιται απο την εκφώνηση μετά το διάβασμα κάθε
            # frame προστίθεται salt&pepper θόρυβος με seed ίσο με
            # το τελευταίο ψηφίο του ΑΜ του πρώτου  μέλους της ομάδας(ΑΜ:03116164 άρα seed=4)
            # και amount που υπολογίζεται απο την συνάρτηση amount(x) που φαίνεται παραπάνω,
            # όπου x   το προτελευταίο ψηφίο του ΑΜ του ίδιου μέλους(δλδ x=6)
            # χρησιμοποιούμε την συνάρτηση random_noise όπως υποδεικνύεται

            another_frame = skimage.util.random_noise(another_frame, mode='s&p', seed=4, amount=snpAmount(6))
            another_frame = np.array( 255*another_frame, dtype='uint8')
            gray = cv2.cvtColor(another_frame, cv2.COLOR_BGR2GRAY)

            # υπολογίζουμε την οπτική ροή με τον αλγόριθμο Lukas Kanade
            next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_param)

            # επιλέγουμε τα σημεία ενδιαφέροντος για την προηγούμενη θέση
            good_old = prev[status == 1]

            # επιλέγουμε τα σημεία ενδιαφέροντος για την επόμενη θέση
            good_new = next[status == 1]

            # ζωγραφίζουμε την οπτική ροή
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                # a,b οι συντεταγμένες του νέου σημείου
                a, b = np.int0(new.ravel())

                # c,d οι συντεταγμένες του παλιού σημείου
                c, d = np.int0(old.ravel())

                # ζωγραφίζουμε την γραμμή μεταξύ της παλιάς και της καινούργιας θέσης
                # με κόκκινο χρώμα και πάχος 3
                # (αύξησαμε το πάχος για να μπορουμε να διακρίνουμε καλυτερα
                # το αποτέλεσμα της οπτικής ροής στην εικόνα με τον θόρυβο)

                mask = cv2.line(mask, (a, b), (c, d), color, 3)

                # ζωγραφίζουμε ένα "γεμάτο" κύκλο στη νέα θέση με ακτίνα 2
                third_frame = cv2.circle(another_frame, (a, b), 2, color, -1)

            # επικαλύπτει τα ίχνη οπτικής ροής στο αρχικό πλαίσιο
            output = cv2.add(another_frame, mask)

            # ενημερώνει το previous frame
            prev_gray = gray.copy()

            # ενημερώνει τα previous good features points
            prev = good_new.reshape(-1, 1, 2)

            # ανοίγουμε νέο παράθυρο  και εμφανίζουμε το αποτέλεσμα
            cv2.imshow("Lukas Kanade_noise", output)

            # για κάθε 20 frames προσθέτουμε μόνο τις γωνίες που μετακινούνται
            if counter == framesForCheck:
                # οταν διαβάσουμε 20 frames , ο μετρητής ξανααρχικοποιείται στο 0
                counter = 0
                prev_gray = gray
                previus_corners = prev
                prev = cv2.goodFeaturesToTrack(prev_gray, **corner_param)

                # ελέγχουμε πάλι ποιες γωνίες κραταμε
                prev = good_corners(prev, starting_corners)
                prev = good_corners(prev, previus_corners)

            # αύξηση μετρητή κατα 1 για να ξέρουμε σε ποιο frame είμαστε
            counter += 1
        else:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Για το ερώτημα 7

def denoise(cap, lk_param, corner_param):
    framesForCheck = 20
    neighborhood = disk(radius=7)

    # αρχικοποιούμε το χρώμα με το οποίο θα ζωγραφιστούν
    # οι γραμμές που θα δείχνουν την οπτική ροή
    color = (0, 0, 255)

    # διαβάζουμε το 20ο frame
    for i in range(framesForCheck):
        reti, test_frame = cap.read()

    # μειώνουμε την ανάλυση στο μισό,όπως απαιτείται
    # απο το 1ο βήμα της άσκησης
    height, width = test_frame.shape[0:2]
    test_frame = cv2.resize(test_frame, (width // 2, height // 2), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    # μετατρέπουμε σε grayscale
    prev_gray2 = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)

    # προσθήκη θορύβου στο 20ο frame
    prev_gray2 = skimage.util.random_noise(prev_gray2, mode='s&p', seed=4, amount=snpAmount(6))
    prev_gray2 = np.array(255 * prev_gray2, dtype='uint8')

    # αποθορυβοποίηση χρησιμοποιώντας median φίλτρο
    prev_gray2 = filters.rank.median(prev_gray2, neighborhood)
    prev_gray2 = np.array(255 * prev_gray2, dtype='uint8')

    # διαβάζουμε το 1ο frame και κάνουμε πάλι το
    # απαραίτητο resize
    ret, first_frame = cap.read()
    height, width = first_frame.shape[0:2]
    first_frame = cv2.resize(first_frame, (width // 2, height // 2), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # προσθήκη θορύβου στο 1ο frame
    prev_gray = skimage.util.random_noise(prev_gray, mode='s&p', seed=4, amount=snpAmount(6))
    prev_gray = np.array(255 * prev_gray, dtype='uint8')

    # αποθορυβοποίηση με median φιλτρο
    prev_gray = filters.rank.median(prev_gray, neighborhood)
    prev_gray = np.array(255 * prev_gray, dtype='uint8')

    # αρχικοποιούμε τον μετρητή που μας δείχνει
    # σε ποιο frame είμαστε κάθε φορά
    counter = 0

    # βρίσκουμε τα σημεία ενδιαφέροντος-γωνίες στο 1ο και στο 20ο frame
    # ενώ έχει προστεθέι θόρυβος
    prev = cv2.goodFeaturesToTrack(prev_gray, **corner_param)
    next2 = cv2.goodFeaturesToTrack(prev_gray2, **corner_param)

    # στην μεταβλητή starting_corners σώζουμε τις γωνίες του 1ου frame
    starting_corners = prev

    # διαγράφουμε τις γωνίες που δεν έχουν αλλάξει θέση απο το 1ο
    # στο 20ο frame , με την χρήση της συνάρτησης good_corners
    # που παρατίθεται παραπάνω
    prev = good_corners(prev, next2)

    # φτιάχνουμε μια μάσκα με μηδενικα, με μέγεθος όσο και το 1ο frame
    mask = np.zeros_like(first_frame)

    while cap.isOpened():

        # διάβασμα και resize του frame
        ret, another_frame = cap.read()
        height, width = another_frame.shape[0:2]
        another_frame = cv2.resize(another_frame, (width // 2, height // 2), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        if ret == True:

            # όπως ζητέιται απο την εκφώνηση μετά το διάβασμα κάθε
            # frame προστίθεται salt&pepper θόρυβος με seed ίσο με
            # το τελευταίο ψηφίο του ΑΜ του πρώτου  μέλους της ομάδας(ΑΜ:03116164 άρα seed=4)
            # και amount που υπολογίζεται απο την συνάρτηση amount(x) που φαίνεται παραπάνω,
            # όπου x   το προτελευταίο ψηφίο του ΑΜ του ίδιου μέλους(δλδ x=6)
            # χρησιμοποιούμε την συνάρτηση random_noise όπως υποδεικνύεται

            gray = cv2.cvtColor(another_frame, cv2.COLOR_BGR2GRAY)

            gray = skimage.util.random_noise(gray, mode='s&p', seed=4, amount=snpAmount(6))
            gray = np.array(255 * gray, dtype='uint8')

            # αποθορυβοποίηση με median φίλτρο
            gray = filters.rank.median(gray, neighborhood)
            gray = np.array(255 * gray, dtype='uint8')

            # υπολογίζουμε την οπτική ροή με τον αλγόριθμο Lukas Kanade
            next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_param)

            # επιλέγουμε τα σημεία ενδιαφέροντος για την προηγούμενη θέση
            good_old = prev[status == 1]

            # επιλέγουμε τα σημεία ενδιαφέροντος για την επόμενη θέση
            good_new = next[status == 1]

            # ζωγραφίζουμε την οπτική ροή
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                # a,b οι συντεταγμένες του νέου σημείου
                a, b = np.int0(new.ravel())

                # c,d οι συντεταγμένες του παλιού σημείου
                c, d = np.int0(old.ravel())

                # ζωγραφίζουμε την γραμμή μεταξύ της παλιάς και της καινούργιας θέσης
                # με κόκκινο χρώμα και πάχος 3
                # (αύξησαμε το πάχος για να μπορουμε να διακρίνουμε καλυτερα
                # το αποτέλεσμα της οπτικής ροής στην εικόνα με τον θόρυβο)

                mask = cv2.line(mask, (a, b), (c, d), color, 3)

                # ζωγραφίζουμε ένα "γεμάτο" κύκλο στη νέα θέση με ακτίνα 2
                third_frame = cv2.circle(another_frame, (a, b), 2, color, -1)

            # επικαλύπτει τα ίχνη οπτικής ροής στο αρχικό πλαίσιο
            output = cv2.add(another_frame, mask)

            # ενημερώνει το previous frame
            prev_gray = gray.copy()

            # ενημερώνει τα previous good features points
            prev = good_new.reshape(-1, 1, 2)

            # ανοίγουμε νέο παράθυρο  και εμφανίζουμε το αποτέλεσμα
            cv2.imshow("Lukas Kanade_denoise", output)

            # για κάθε 20 frames προσθέτουμε μόνο τις γωνίες που μετακινούνται
            if counter == framesForCheck:
                # οταν διαβάσουμε 20 frames , ο μετρητής ξανααρχικοποιείται στο 0
                counter = 0
                prev_gray = gray
                previus_corners = prev
                prev = cv2.goodFeaturesToTrack(prev_gray, **corner_param)

                # ελέγχουμε πάλι ποιες γωνίες κραταμε
                prev = good_corners(prev, starting_corners)
                prev = good_corners(prev, previus_corners)

            # αύξηση μετρητή κατα 1 για να ξέρουμε σε ποιο frame είμαστε
            counter += 1
        else:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Ερώτημα 1

# διαβάζουμε το πρώτο frame και
# ρίχνουμε την ανάλυση στο μισό

cap = cv2.VideoCapture("C:\dromos.mp4")

ret, first_frame = cap.read()
height, width = first_frame.shape[0:2]
first_frame = cv2.resize(first_frame, (width // 2, height // 2), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

# Ερώτημα 2
# μετατρέπουμε σε grayscale
ff_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)



# Ερώτημα 3
# εφαρμογή shi_tomasi
# πειραματισμός με παραμέτρους

# δοκιμή 1

# παράμετροι για εντοπισμό shi tomasi corners

feature_params1 = dict(maxCorners=400, qualityLevel=0.2, minDistance=2, blockSize=7)

shi_tomasi1 = cv2.goodFeaturesToTrack(ff_gray.copy(), mask=None, **feature_params1)
shi_tomasi_count1 = np.int0(shi_tomasi1)

first_frame1 = copy.copy(first_frame)

# αποτυπώνουμε τα shi tomasi corners με γεμάτους κύκλους ακτίνας 2
for i in shi_tomasi_count1:
    x, y = i.ravel()
    cv2.circle(first_frame1, (x, y), 2, 255, -1)
# εμφανίζουμε τα αποτελέσματα
plt.imshow(first_frame1), plt.title("maxCorners=400, qualityLevel=0.2, minDistance=2"), plt.show()

# δοκιμή 2

feature_params2 = dict(maxCorners=800, qualityLevel=0.2, minDistance=2, blockSize=7)

shi_tomasi2 = cv2.goodFeaturesToTrack(ff_gray.copy(), mask=None, **feature_params2)
shi_tomasi_count2 = np.int0(shi_tomasi2)

first_frame2 = copy.copy(first_frame)

for i in shi_tomasi_count2:
    x, y = i.ravel()
    cv2.circle(first_frame2, (x, y), 2, 255, -1)

plt.imshow(first_frame2), plt.title("maxCorners=800, qualityLevel=0.2, minDistance=2"), plt.show()

# δοκιμή 3

feature_params3 = dict(maxCorners=800, qualityLevel=0.05, minDistance=2, blockSize=7)

shi_tomasi3 = cv2.goodFeaturesToTrack(ff_gray.copy(), mask=None, **feature_params3)
shi_tomasi_count3 = np.int0(shi_tomasi3)

first_frame3 = copy.copy(first_frame)

for i in shi_tomasi_count3:
    x, y = i.ravel()
    cv2.circle(first_frame3, (x, y), 2, 255, -1)

plt.imshow(first_frame3), plt.title("maxCorners=800, qualityLevel=0.05, minDistance=2"), plt.show()

# δοκιμή 4
feature_params4 = dict(maxCorners=800, qualityLevel=0.05, minDistance=3, blockSize=7)

shi_tomasi4 = cv2.goodFeaturesToTrack(ff_gray.copy(), mask=None, **feature_params4)
shi_tomasi_count4 = np.int0(shi_tomasi4)

first_frame4 = copy.copy(first_frame)

for i in shi_tomasi_count4:
    x, y = i.ravel()
    cv2.circle(first_frame4, (x, y), 2, 255, -1)

plt.imshow(first_frame4), plt.title("maxCorners=800, qualityLevel=0.05, minDistance=3"), plt.show()

# δοκιμή 5
feature_params5 = dict(maxCorners=800, qualityLevel=0.05, minDistance=5, blockSize=7)

shi_tomasi5 = cv2.goodFeaturesToTrack(ff_gray.copy(), mask=None, **feature_params5)
shi_tomasi_count5 = np.int0(shi_tomasi4)

first_frame5 = copy.copy(first_frame)

for i in shi_tomasi_count5:
    x, y = i.ravel()
    cv2.circle(first_frame5, (x, y), 2, 255, -1)

plt.imshow(first_frame5), plt.title("maxCorners=800, qualityLevel=0.05, minDistance=5"), plt.show()

# εφαρμογή harris corner

# δοκιμή 1
feature_param1 = dict(maxCorners=400, qualityLevel=0.1, minDistance=2, blockSize=7, useHarrisDetector=True)

harris1 = cv2.goodFeaturesToTrack(ff_gray, mask=None, **feature_param1)
harris_count1 = np.int0(harris1)

first_fram1 = copy.copy(first_frame)

for i in harris_count1:
    x, y = i.ravel()
    cv2.circle(first_fram1, (x, y), 2, 255, -1)

plt.imshow(first_fram1), plt.title("maxCorners=400, qualityLevel=0.1, minDistance=2"), plt.show()

# δοκιμή 2
feature_param2 = dict(maxCorners=800, qualityLevel=0.1, minDistance=2, blockSize=7, useHarrisDetector=True)

harris2 = cv2.goodFeaturesToTrack(ff_gray, mask=None, **feature_param2)
harris_count2 = np.int0(harris2)

first_fram2 = copy.copy(first_frame)

for i in harris_count2:
    x, y = i.ravel()
    cv2.circle(first_fram2, (x, y), 2, 255, -1)

plt.imshow(first_fram2), plt.title("maxCorners=800, qualityLevel=0.1, minDistance=2"), plt.show()

# δοκιμή 3

feature_param3 = dict(maxCorners=800, qualityLevel=0.05, minDistance=2, blockSize=7, useHarrisDetector=True)

harris3 = cv2.goodFeaturesToTrack(ff_gray, mask=None, **feature_param3)
harris_count3 = np.int0(harris3)

first_fram3 = copy.copy(first_frame)

for i in harris_count3:
    x, y = i.ravel()
    cv2.circle(first_fram3, (x, y), 2, 255, -1)

plt.imshow(first_fram3), plt.title("maxCorners=800, qualityLevel=0.05, minDistance=2"), plt.show()

# δοκιμή 4
feature_param4 = dict(maxCorners=800, qualityLevel=0.05, minDistance=4, blockSize=7, useHarrisDetector=True)

harris4 = cv2.goodFeaturesToTrack(ff_gray, mask=None, **feature_param4)
harris_count4 = np.int0(harris4)

first_fram4 = copy.copy(first_frame)

for i in harris_count4:
    x, y = i.ravel()
    cv2.circle(first_fram4, (x, y), 2, 255, -1)

plt.imshow(first_fram4), plt.title("maxCorners=800, qualityLevel=0.05, minDistance=4"), plt.show()

# δοκιμή 5
feature_param5 = dict(maxCorners=1200, qualityLevel=0.02, minDistance=5, blockSize=7, useHarrisDetector=True)

harris5 = cv2.goodFeaturesToTrack(ff_gray, mask=None, **feature_param5)
harris_count5 = np.int0(harris5)

first_fram5 = copy.copy(first_frame)

for i in harris_count5:
    x, y = i.ravel()
    cv2.circle(first_fram5, (x, y), 2, 255, -1)

plt.imshow(first_fram5), plt.title("maxCorners=1200, qualityLevel=0.02, minDistance=5"), plt.show()

# Ερώτημα 4

# Lukas Kanade για όλες τις δοκιμές shi tomasi

Lucas_Kanade(cv2.VideoCapture("C:\dromos.mp4"), shi_tomasi1, (15, 15), 2,
             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
Lucas_Kanade(cv2.VideoCapture("C:\dromos.mp4"), shi_tomasi2, (15, 15), 2,
             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
Lucas_Kanade(cv2.VideoCapture("C:\dromos.mp4"), shi_tomasi3, (15, 15), 2,
             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
Lucas_Kanade(cv2.VideoCapture("C:\dromos.mp4"), shi_tomasi4, (15, 15), 2,
             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
Lucas_Kanade(cv2.VideoCapture("C:\dromos.mp4"), shi_tomasi5, (15, 15), 2,
             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Lukas Kanade για όλες τις δοκιμές harris corner

Lucas_Kanade(cv2.VideoCapture("C:\dromos.mp4"), harris1, (15, 15), 2,
             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
Lucas_Kanade(cv2.VideoCapture("C:\dromos.mp4"), harris2, (15, 15), 2,
             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
Lucas_Kanade(cv2.VideoCapture("C:\dromos.mp4"), harris3, (15, 15), 2,
             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
Lucas_Kanade(cv2.VideoCapture("C:\dromos.mp4"), harris4, (15, 15), 2,
             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
Lucas_Kanade(cv2.VideoCapture("C:\dromos.mp4"), harris5, (15, 15), 2,
             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Ερώτημα 5
# σε αυτό το ερώτημα χρησιμοποιούμε μόνο τα best r
# των ερωτημάτων 3,4

# θέτουμε τα ορίσματα της συνάρτησης
lk_param = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# καλούμε την συνάρτηση
# για το καλύτερο αποτέλεσμα για shi_tomasi
lucas_kanade_5(cv2.VideoCapture("C:\dromos.mp4"), lk_param,feature_params5 )

# καλούμε την συνάρτηση
# για το καλύτερο αποτέλεσμα για harris corner
lucas_kanade_5(cv2.VideoCapture("C:\dromos.mp4"), lk_param,feature_param5 )

# Ερώτημα 6

# καλούμε την συνάρτηση
# για το καλύτερο αποτέλεσμα για shi_tomasi
lucas_kanade_noise(cv2.VideoCapture("C:\dromos.mp4"), lk_param,feature_params5 )

# καλούμε την συνάρτηση
# για το καλύτερο αποτέλεσμα για harris corner
lucas_kanade_noise(cv2.VideoCapture("C:\dromos.mp4"), lk_param,feature_param5 )


# Ερώτημα 7
denoise(cv2.VideoCapture("C:\dromos.mp4"), lk_param,feature_params5 )
denoise(cv2.VideoCapture("C:\dromos.mp4"), lk_param,feature_params5 )

