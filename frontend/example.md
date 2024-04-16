# Example provided by Nicklas
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>API Requests Example</title>
    <script>
      document.addEventListener("DOMContentLoaded", function () {
        // Api key for local development
        // const apiKey = "a6bf09ec-09a5-4e2d-a142-43e77e87b3d0";
        // Api key for access to public api at bandim.infonest.xyz
        const apiKey =
          "sp7b0JE3tg6RnhFRVsLVE1Vdsv2o7Riu12qlPHbcAOZPit30PdtIPWNi7gFDOahAB6GWilxWXik7qe5i4CLytIDaUwDOp4LnP00ZrdHS34YqKnXQAfp15TjQjWC3AHGP";

        // Reference to the elements in the HTML
        const createUserButton = document.getElementById("createUser");
        const loginButton = document.getElementById("login");
        const fetchUserDataButton = document.getElementById("fetchUserData");
        const responseDisplay = document.getElementById("responseDisplay");

        // Temporary storage for the access token
        let accessToken = "";

        // Function to create a user
        async function createUser() {
          const username = "newuser1"; // should use form input
          const password = "password123!"; // should use form input
          try {
            const response = await fetch(
              `https://bandim.infonest.xyz/api/public/auth/register`,
              {
                method: "POST",
                mode: "cors",
                headers: {
                  "Content-Type": "application/json",
                  "X-API-Key": `${apiKey}`,
                },
                body: JSON.stringify({ username, password }),
              }
            );
            const data = await response.json();
            responseDisplay.textContent = JSON.stringify(data, null, 2);
          } catch (error) {
            responseDisplay.textContent =
              "Failed to create user: " + error.message;
          }
        }

        // Function to log in and get access and refresh tokens
        async function loginAndGetTokens() {
          const username = "newuser1"; // should use form input
          const password = "password123!"; // should use form input
          try {
            const response = await fetch(
              `https://bandim.infonest.xyz/api/public/auth/login`,
              {
                method: "POST",
                mode: "cors",
                headers: {
                  "Content-Type": "application/json",
                  "X-API-Key": `${apiKey}`,
                },
                body: JSON.stringify({ username, password }),
              }
            );
            const data = await response.json();
            accessToken = data.access_token; // Save the access token
            responseDisplay.textContent = JSON.stringify(data, null, 2);
          } catch (error) {
            responseDisplay.textContent = "Login failed: " + error.message;
          }
        }

        // Function to get user account data using the access token
        async function getUserAccountData() {
          try {
            const response = await fetch(
              `https://bandim.infonest.xyz/api/public/users/me`,
              {
                method: "GET",
                mode: "cors",
                headers: {
                  "Content-Type": "application/json",
                  Authorization: `Bearer ${accessToken}`,
                  "X-API-Key": `${apiKey}`,
                },
              }
            );
            const data = await response.json();
            responseDisplay.textContent = JSON.stringify(data, null, 2);
          } catch (error) {
            responseDisplay.textContent =
              "Failed to get user account data: " + error.message;
          }
        }

        // Event listeners for the buttons
        createUserButton.addEventListener("click", createUser);
        loginButton.addEventListener("click", loginAndGetTokens);
        fetchUserDataButton.addEventListener("click", getUserAccountData);
      });
    </script>
  </head>
  <body>
    <h2>API Request Example</h2>
    <button id="createUser">Create User</button>
    <button id="login">Login</button>
    <button id="fetchUserData">Fetch User Data</button>
    <pre id="responseDisplay"></pre>
  </body>
</html>
```