#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hra střelnice
"""

import pygame
import argparse
import ConfigParser
import sys
import json


class ShootingGallery():

    def __init__(self):
        """
        Inicializační funkce. Volá se jen jednou na začátku.
        """
        pass

    def tick(self):
        """
        Tato funkce se vykonává opakovaně
        """
        pass

    def run(self):
        """
        funkce opakovaně volá funkci tick
        """
        #  Ošetření vstupních událostí
        for udalost in pygame.event.get():
            if udalost.type == pygame.locals.QUIT:
                return
            elif (udalost.type == pygame.locals.KEYDOWN and
                  udalost.key == pygame.locals.K_ESCAPE):
                return
        casovac = pygame.time.Clock()
        while 1:
            casovac.tick(20)
            self.tick()


def main():
    args = get_params()
    print vars(args)
    # convert to ints
    print json.loads(args.target_center)
    sh = ShootingGallery()
    sh.run()


def get_params(argv=None):
    """
    Funkce načte parametry z příkazové řádky nebo z konfiguračního souboru.
    """

    # načítání konfigurace
    # Do argv default this way, as doing it in the functional
    # declaration sets it at compile time.
    if argv is None:
        argv = sys.argv

    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
    )
    conf_parser.add_argument("-c", "--conf_file",
                             help="Specify config file", metavar="FILE",
                             default='config')
    args, remaining_argv = conf_parser.parse_known_args()

    if args.conf_file:
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        defaults = dict(config.items("Defaults"))
    else:
        defaults = {"option": "default"}

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser]
    )
    parser.set_defaults(**defaults)
    parser.add_argument("--option")
    args = parser.parse_args(remaining_argv)
    # v args jsou teď všechny parametry
    return args


if __name__ == "__main__":
    main()
