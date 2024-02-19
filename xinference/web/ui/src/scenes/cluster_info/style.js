import { createTheme, tableCellClasses } from '@mui/material'
import Paper from '@mui/material/Paper'
import TableCell from '@mui/material/TableCell'
import TableRow from '@mui/material/TableRow'
import { styled } from '@mui/system'

export const theme = createTheme()

// export const useStyles = makeStyles((theme) => ({
//   root: {
//     display: 'flex',
//   },
//   menuButton: {
//     marginRight: 36,
//   },
//   menuButtonHidden: {
//     display: 'none',
//   },
//   menuButtonNested: {
//     paddingLeft: theme.spacing(4),
//   },
//   leftMenu: {
//     height: '100%',
//   },
//   leftMenuBottomItem: {
//     display: 'flex',
//     flexDirection: 'row',
//     justifyContent: 'flex-end',
//   },
//   nestedListItem: {
//     paddingLeft: theme.spacing(4),
//   },
//   appBarSpacer: theme.mixins.toolbar,
//   content: {
//     flexGrow: 1,
//     height: '100vh',
//     overflow: 'auto',
//   },
//   container: {
//     paddingTop: theme.spacing(4),
//     paddingBottom: theme.spacing(4),
//   },
//   paper: {
//     padding: theme.spacing(2),
//     display: 'flex',
//     overflow: 'auto',
//     flexDirection: 'column',
//   },
//   fixedHeight: {
//     height: 240,
//   },
//   logo: {
//     maxWidth: 200,
//     marginRight: 2,
//     padding: 0,
//   },
// }))

export const StyledTableCell = styled(TableCell)(() => ({
  [`&.${tableCellClasses.body}`]: {
    fontSize: 14,
  },
}))

export const StyledTableRow = styled(TableRow)(({ theme }) => ({
  '&:nth-of-type(odd)': {
    backgroundColor: theme.palette.action.hover,
  },
  // hide last border
  '&:last-child td, &:last-child th': {
    border: 0,
  },
}))

export const StyledPaper = styled(Paper)({
  padding: theme.spacing(2),
  display: 'flex',
  overflow: 'auto',
  flexDirection: 'column',
})
