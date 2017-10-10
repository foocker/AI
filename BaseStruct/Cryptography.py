class CaesarCipher:
    ''' Class for doing encryption and decryption using a Caesar cipher.'''

    def __init__(self, shift):
        encoder = [None]*26
        decoder = [None]*26
        for k in range(26):
            encoder[k] = chr((k + shift) % 26 + ord('A'))
            decoder[k] = chr((k - shift) % 26 + ord('A'))
        self._forward = ''.join(encoder)
        self._backward = ''.join(decoder)

    def encrypt(self, message):
        return self._transform(message, self._forward)

    def decrypt(self, secret):
        return self._transform(secret, self._backward)

    def _transform(self, original, code):
        '''Utility to perform transformation based on given code string.'''
        msg = list(original)
        for k in range(len(msg)):
            if msg[k].isupper():
                j = ord(msg[k]) - ord('A')
                msg[k] = code[j]
        return ''.join(msg)

if __name__ == '__main__':
    cipher = CaesarCipher(3)
    message = "I WILL GO HOME, MAY NOT COME BACK, MAY CAN'T SEE YOU AGAIN, BUT WHERE IS MY HOME?"
    coded = cipher.encrypt(message)
    print('Secret:', coded)
    answer = cipher.decrypt(coded)
    print('Message:', answer)
