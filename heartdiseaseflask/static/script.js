// Object to hold existing accounts
let existingAccounts = {};

// Validate email format
function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

// Validate password format
function isValidPassword(password) {
    return password.length >= 8 &&
        /[0-9]/.test(password) &&
        /[a-z]/.test(password) &&
        /[A-Z]/.test(password) &&
        /[!@#$%^&*]/.test(password);
}

// Toggle between login and signup forms
document.querySelector('.login-btn').addEventListener('click', function () {
    document.querySelector('.login-form').style.display = 'block';
    document.querySelector('.signup-container').style.display = 'none';
});

document.querySelector('.signup-btn').addEventListener('click', function () {
    document.querySelector('.signup-container').style.display = 'block';
    document.querySelector('.login-form').style.display = 'none';
});

// Handle user signup
document.querySelector('#user-signup-form').addEventListener('submit', function (e) {
    e.preventDefault();
    let name = document.getElementById('signup-name').value;
    let email = document.getElementById('signup-email').value;
    let password = document.getElementById('signup-password').value;

    // Check if email already exists
    if (existingAccounts[email]) {
        alert('Account already exists! Please login.');
        document.querySelector('.signup-container').style.display = 'none';
        document.querySelector('.login-form').style.display = 'block';
        return;
    }

    // Validate email and password
    if (!validateEmail(email)) {
        alert('Invalid email format.');
        return;
    }

    if (!isValidPassword(password)) {
        alert('Password must contain at least 8 characters, including numbers, uppercase, lowercase letters, and special characters.');
        return;
    }

    // Create new account
    existingAccounts[email] = { name: name, password: password };
    alert('Account created successfully! Redirecting to the login page.');

    // Redirect to login page
    document.querySelector('.signup-container').style.display = 'none';
    document.querySelector('.login-form').style.display = 'block';
});

// Handle user login
document.querySelector('#user-login-form').addEventListener('submit', function (e) {
    e.preventDefault();
    let email = document.querySelector('#user-login-form input[type="email"]').value;
    let password = document.querySelector('#user-login-form input[type="password"]').value;

    // Validate email format
    if (!validateEmail(email)) {
        alert('Please enter a valid email address.');
        return;
    }

    // Check if account exists
    if (!existingAccounts[email]) {
        alert('Account does not exist! Redirecting to signup page.');
        document.querySelector('.login-form').style.display = 'none';
        document.querySelector('.signup-container').style.display = 'block'; // Redirect to signup
        return;
    }

    // Check for incorrect password
    if (existingAccounts[email].password !== password) {
        alert('Incorrect password. Please try again.');
        return;
    }

    // Simulate successful login and redirect to homepage
    alert(`Login successful! Redirecting to home, ${existingAccounts[email].name}.`);
    window.localStorage.setItem('username', existingAccounts[email].name); // Store name in local storage
    window.location.href = 'Homepage.html'; // Redirect to homepage
});

// Handle admin login
document.querySelector('#admin-login-form').addEventListener('submit', function (e) {
    e.preventDefault();
    let email = document.querySelector('#admin-login-form input[type="email"]').value;
    let password = document.querySelector('#admin-login-form input[type="password"]').value;

    // Validate email format
    if (!validateEmail(email)) {
        alert('Please enter a valid email address.');
        return;
    }

    // Check if account exists
    if (!existingAccounts[email]) {
        alert('Account does not exist! Redirecting to signup page.');
        document.querySelector('.login-form').style.display = 'none';
        document.querySelector('.signup-container').style.display = 'block'; // Redirect to signup
        return;
    }

    // Check for incorrect password
    if (existingAccounts[email].password !== password) {
        alert('Incorrect password. Please try again.');
        return;
    }

    // Simulate successful login and redirect to homepage
    alert(`Admin login successful! Redirecting to home, ${existingAccounts[email].name}.`);
    window.localStorage.setItem('username', existingAccounts[email].name); // Store name in local storage
    window.location.href = 'Homepage.html'; // Redirect to homepage
});

// Toggle between user and admin login forms
document.querySelector('.user-login-btn').addEventListener('click', function () {
    document.querySelector('#user-login-form').style.display = 'block';
    document.querySelector('#admin-login-form').style.display = 'none';
    this.classList.add('active');
    document.querySelector('.admin-login-btn').classList.remove('active');
});

document.querySelector('.admin-login-btn').addEventListener('click', function () {
    document.querySelector('#admin-login-form').style.display = 'block';
    document.querySelector('#user-login-form').style.display = 'none';
    this.classList.add('active');
    document.querySelector('.user-login-btn').classList.remove('active');
});

// Toggle between user and admin signup forms
document.querySelector('.user-signup-btn').addEventListener('click', function () {
    document.querySelector('#user-signup-form').style.display = 'block';
    document.querySelector('#admin-signup-form').style.display = 'none';
    this.classList.add('active');
    document.querySelector('.admin-signup-btn').classList.remove('active');
});

document.querySelector('.admin-signup-btn').addEventListener('click', function () {
    document.querySelector('#admin-signup-form').style.display = 'block';
    document.querySelector('#user-signup-form').style.display = 'none';
    this.classList.add('active');
    document.querySelector('.user-signup-btn').classList.remove('active');
});
