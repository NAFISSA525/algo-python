

import numpy as np
import scipy.optimize as opt
import scipy.linalg as la
import scipy.integrate as integrate

# 1. DICHOTOMIE - Méthode de bissection pour trouver les racines
# Utilise scipy.optimize.bisect qui implémente l'algorithme de dichotomie
def dichotomie(fonction, borne_inf, borne_sup, seuil=1e-6):
    """
    Résout f(x) = 0 par la méthode de dichotomie
    Exemple: dichotomie(lambda x: x**2 - 4, 1, 3) → trouve la racine 2.0
    """
    return opt.bisect(fonction, borne_inf, borne_sup, xtol=seuil)

# 2. BALAYAGE - Parcourt l'intervalle avec un pas pour détecter le changement de signe
# Utilise numpy pour l'évaluation vectorisée des fonctions
def balayage(fonction, borne_inf, borne_sup, n_points=1000):
    """
    Résout f(x) = 0 par balayage de l'intervalle
    Exemple: balayage(lambda x: x**2 - 4, 1, 3) → trouve la racine 2.0
    """
    # Crée des points régulièrement espacés dans l'intervalle
    x_points = np.linspace(borne_inf, borne_sup, n_points)
    # Évalue la fonction sur tous les points
    y_points = fonction(x_points)
    # Trouve l'indice où la fonction change de signe
    idx_changement = np.where(np.diff(np.sign(y_points)))[0][0]
    # Retourne le milieu du segment où le changement se produit
    return (x_points[idx_changement] + x_points[idx_changement + 1]) / 2

# 3. LAGRANGE - Méthode de la fausse position (Regula Falsi)
# Utilise scipy.optimize.brentq qui combine dichotomie et interpolation
def lagrange(fonction, borne_inf, borne_sup, seuil=1e-6):
    """
    Résout f(x) = 0 par la méthode de Lagrange (régula falsi)
    Exemple: lagrange(lambda x: x**2 - 4, 1, 3) → trouve la racine 2.0
    """
    return opt.brentq(fonction, borne_inf, borne_sup, xtol=seuil)

# 4. NEWTON-RAPHSON - Utilise la dérivée pour convergence rapide
# Utilise scipy.optimize.newton avec la dérivée fournie
def newton_raphson(fonction, derivee, point_initial, seuil=1e-6):
    """
    Résout f(x) = 0 par la méthode de Newton-Raphson
    Exemple: newton_raphson(lambda x: x**2 - 4, lambda x: 2*x, 2.5) → 2.0
    """
    return opt.newton(fonction, point_initial, fprime=derivee, tol=seuil)

# 5. PIVOT DE GAUSS - Résolution de systèmes linéaires par élimination
# Utilise numpy.linalg.solve qui implémente l'élimination de Gauss
def pivot_gauss(matrice, vecteur):
    """
    Résout le système Ax = b par la méthode du pivot de Gauss
    Exemple: A = [[2,1],[1,3]], b = [5,10] → solution [1, 3]
    """
    return np.linalg.solve(matrice, vecteur)

# 6. GAUSS-JORDAN - Résolution de systèmes linéaires
# Utilise scipy.linalg.solve (méthode directe optimisée)
def gauss_jordan(matrice, vecteur):
    """
    Résout le système Ax = b (équivalent à Gauss-Jordan)
    Exemple: A = [[2,1],[1,3]], b = [5,10] → solution [1, 3]
    """
    return la.solve(matrice, vecteur)

# 7. CROUT - Décomposition LU pour résoudre les systèmes
# Utilise scipy.linalg.lu_factor et lu_solve
def methode_crout(matrice, vecteur):
    """
    Résout Ax = b par décomposition LU (méthode de Crout)
    Exemple: A = [[2,1],[1,3]], b = [5,10] → solution [1, 3]
    """
    # Factorise la matrice A en matrices L et U
    decomposition_lu = la.lu_factor(matrice)
    # Résout le système en utilisant la décomposition LU
    return la.lu_solve(decomposition_lu, vecteur)



def newton_cote(fonction, borne_inf, borne_sup, methode='simpson', n_points=1000):
    """
    Calcule l'intégrale numérique par les méthodes de Newton-Côte
    
    Args:
        fonction: Fonction à intégrer
        borne_inf: Borne inférieure d'intégration
        borne_sup: Borne supérieure d'intégration
        methode: 'trapeze', 'simpson', 'simpson_3_8', 'boole'
        n_points: Nombre de points d'évaluation
        
    Returns:
        Valeur approximative de l'intégrale
        
    Exemple:
        newton_cote(lambda x: x**2, 0, 2) → ≈2.666 (8/3)
        newton_cote(lambda x: np.sin(x), 0, np.pi) → ≈2.0
    """
    x_points = np.linspace(borne_inf, borne_sup, n_points)
    y_points = fonction(x_points)
    
    if methode == 'trapeze':
        # Méthode des trapèzes (n=2) - Formule fermée
        return integrate.trapezoid(y_points, x_points)
    
    elif methode == 'simpson':
        # Méthode de Simpson (n=3) - Formule 1/3
        return integrate.simpson(y_points, x_points)
    
    elif methode == 'simpson_3_8':
        # Méthode de Simpson 3/8 (n=4)
        # Pour 3/8 rule, on besoin de (3k+1) points
        n = len(x_points)
        if (n - 1) % 3 != 0:
            n = ((n - 1) // 3) * 3 + 1
            x_points = np.linspace(borne_inf, borne_sup, n)
            y_points = fonction(x_points)
        return integrate.simpson(y_points, x_points)
    
    elif methode == 'boole':
        # Méthode de Boole (n=5) - Formule de degré 4
        return _methode_boole(fonction, borne_inf, borne_sup, n_points)
    
    else:
        raise ValueError(f"Méthode {methode} non reconnue. Choisir: 'trapeze', 'simpson', 'simpson_3_8', 'boole'")

def _methode_boole(f, a, b, n_points):
    """Implémentation de la méthode de Boole (n=5 points)"""
    # S'assure d'avoir un multiple de 4 intervalles (5 points par segment)
    n = ((n_points - 1) // 4) * 4 + 1
    x = np.linspace(a, b, n)
    h = (b - a) / (n - 1)
    
    # Formule de Boole: (2h/45)[7f0 + 32f1 + 12f2 + 32f3 + 7f4] pour chaque segment
    resultat = 0
    for i in range(0, n-1, 4):
        resultat += 7*f(x[i]) + 32*f(x[i+1]) + 12*f(x[i+2]) + 32*f(x[i+3]) + 7*f(x[i+4])
    
    return (2 * h / 45) * resultat

# EXEMPLE D'UTILISATION
if __name__ == "__main__":
    # Test avec x² sur [0,2] - intégrale exacte = 8/3 ≈ 2.6667
    resultat = newton_cote(lambda x: x**2, 0, 2, 'simpson')
    print(f"Intégrale de x² sur [0,2]: {resultat}")
    print(f"Valeur exacte: 8/3 = {8/3}")
    
    # Test avec sin(x) sur [0,π] - intégrale exacte = 2.0
    resultat2 = newton_cote(np.sin, 0, np.pi, 'simpson')
    print(f"Intégrale de sin(x) sur [0,π]: {resultat2}")
    print(f"Valeur exacte: 2.0")
    
    
    
# EXEMPLE D'UTILISATION COMPLET
if __name__ == "__main__":
    print("=== EXEMPLES D'UTILISATION DES MÉTHODES NUMÉRIQUES ===\n")
    
    # Définition des fonctions de test
    def f(x):
        """Fonction test: x² - 4 = 0, racine en x = 2"""
        return x**2 - 4
    
    def f_prime(x):
        """Dérivée de la fonction test: 2x"""
        return 2*x
    
    # Système linéaire test: 2x + y = 5, x + 3y = 10
    # Solution: x = 1, y = 3
    A = np.array([[2, 1], 
                  [1, 3]], dtype=float)
    b = np.array([5, 10], dtype=float)
    
    print("1. DICHOTOMIE - Résolution de x² - 4 = 0 sur [1, 3]")
    racine_dicho = dichotomie(f, 1, 3)
    print(f"   Racine trouvée: {racine_dicho}")
    print(f"   Vérification: f({racine_dicho}) = {f(racine_dicho)}\n")
    
    print("2. BALAYAGE - Résolution de x² - 4 = 0 sur [1, 3]")
    racine_balayage = balayage(f, 1, 3)
    print(f"   Racine trouvée: {racine_balayage}")
    print(f"   Vérification: f({racine_balayage}) = {f(racine_balayage)}\n")
    
    print("3. LAGRANGE - Résolution de x² - 4 = 0 sur [1, 3]")
    racine_lagrange = lagrange(f, 1, 3)
    print(f"   Racine trouvée: {racine_lagrange}")
    print(f"   Vérification: f({racine_lagrange}) = {f(racine_lagrange)}\n")
    
    print("4. NEWTON-RAPHSON - Résolution de x² - 4 = 0 avec x0 = 2.5")
    racine_newton = newton_raphson(f, f_prime, 2.5)
    print(f"   Racine trouvée: {racine_newton}")
    print(f"   Vérification: f({racine_newton}) = {f(racine_newton)}\n")
    
    print("5. PIVOT DE GAUSS - Résolution du système:")
    print("   2x + y = 5")
    print("   x + 3y = 10")
    solution_gauss = pivot_gauss(A, b)
    print(f"   Solution: x = {solution_gauss[0]:.1f}, y = {solution_gauss[1]:.1f}")
    print(f"   Vérification: A @ solution = {A @ solution_gauss}\n")
    
    print("6. GAUSS-JORDAN - Résolution du même système")
    solution_gj = gauss_jordan(A, b)
    print(f"   Solution: x = {solution_gj[0]:.1f}, y = {solution_gj[1]:.1f}")
    print(f"   Vérification: A @ solution = {A @ solution_gj}\n")
    
    print("7. MÉTHODE DE CROUT - Résolution par décomposition LU")
    solution_crout = methode_crout(A, b)
    print(f"   Solution: x = {solution_crout[0]:.1f}, y = {solution_crout[1]:.1f}")
    print(f"   Vérification: A @ solution = {A @ solution_crout}")