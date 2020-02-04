# Issues

This guide is not yet drafted. Visit the [legacy documentation](https://github.com/econ-ark/HARK/tree/master/Documentation) for more info.


## Linking a fork to your local clone.

NOTE: Use this guide if you have cloned HARK locally by doing `git clone https://github.com/econ-ark/HARK`.

If you want to submit the changes that you have done locally and get the changes merged into the HARK repository you need to add a link to your `fork` of the project so that you can create a pull request. A pull request is required to review your code and merge it with the HARK codebase.

1. Create a fork by navigating to https://github.com/econ-ark/HARK and click on `Fork` (top right corner). Select your username on GitHub under the account heading (if there is a pop-up).

2. Locally on your machine, navigate to the HARK directory.
```
$ cd /path/to/HARK
```

3. Check the current `remote` links of your local copy of HARK.

It should look something like this.
```
$ git remote -v
origin        https://github.com/econ-ark/HARK (fetch)
origin        https://github.com/econ-ark/HARK (push)
```

4. Add a new `remote` which adds link to your fork.

Change `user_name` to your `user_name` on GitHub before executing the following command.
```
$ git remote add user_name https://github.com/user_name/HARK
```

If we check the remote of this repository again it should link towards your fork too.

```
$ git remote -v
origin        https://github.com/econ-ark/HARK (fetch)
origin        https://github.com/econ-ark/HARK (push)
user_name    https://github.com/user_name/HARK (fetch)
user_name    https://github.com/user_name/HARK (push)
```

5. Now that you have a link to your fork you can start creating Pull Requests.
